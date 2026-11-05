"""Independent, bounded sources. Raw records and metadata remain inspectable."""
import itertools
import json
import logging
import zipfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

from src.common import digest, path, write_json, write_jsonl

LOG = logging.getLogger(__name__)


def hf_source(spec):
    from datasets import get_dataset_config_names, load_dataset
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(spec["repo"], revision=spec.get("revision", "main"))
    card = info.card_data.to_dict() if info.card_data else {}
    names = get_dataset_config_names(spec["repo"], revision=info.sha)
    name = spec.get("config") or (names[0] if len(names) == 1 else None)
    if name is None:
        raise ValueError(f"{spec['id']}: choose config from {names}")
    dataset = load_dataset(spec["repo"], name=name, split=spec.get("split", "train"),
                           revision=info.sha, streaming=True)
    metadata = {"repo": spec["repo"], "revision": info.sha, "config": name,
                "split": spec.get("split", "train"), "card": card,
                "features": str(dataset.features), "license": card.get("license", "unknown")}

    def records():
        used = 0
        for i, row in enumerate(itertools.islice(dataset, spec["limit"])):
            payload = json.dumps(row, ensure_ascii=False)
            metadata.setdefault("observed_columns", {k:type(v).__name__ for k,v in row.items()})
            used += len(payload.encode("utf-8"))
            if used > spec["max_bytes"]:
                break
            yield {"raw": row, "source": spec["repo"], "source_revision": info.sha,
                   "source_config": name, "source_split": metadata["split"],
                   "document_id": digest(f"{spec['repo']}:{info.sha}:{name}:{metadata['split']}:{i}"),
                   "license": metadata["license"], "group": spec["group"]}
    return metadata, records()


def bounded_get(url, budget):
    import requests
    with requests.get(url, timeout=(10, 30), stream=True) as response:
        response.raise_for_status()
        body = bytearray()
        for chunk in response.iter_content(16384):
            body.extend(chunk)
            if len(body) > budget:
                raise ValueError(f"Response exceeds {budget} bytes: {url}")
        # requests falls back to ISO-8859-1 for HTML; old ae-lib pages may be cp1251.
        from bs4 import UnicodeDammit
        return UnicodeDammit(bytes(body), is_html=True).unicode_markup or ""


def web_source(spec):
    from bs4 import BeautifulSoup
    index = bounded_get(spec["url"], spec["max_bytes"])
    soup = BeautifulSoup(index, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        url = urljoin(spec["url"], a["href"])
        if (urlparse(url).netloc == urlparse(spec["url"]).netloc and
                urlparse(url).path.endswith((".htm", ".html", ".txt")) and
                url != spec["url"] and url not in links):
            links.append(url)
    def records():
        for url in links[:spec["limit"]]:
            try:
                text = bounded_get(url, spec["max_bytes"])
                yield {"raw": {"text": text}, "source": url, "group": spec["group"],
                       "license": "unknown", "document_id": digest(url + text)}
            except Exception as e:
                LOG.warning("Cannot import %s: %s", url, e)
    return {"url": spec["url"], "links_found": len(links), "license": "unknown"}, records()


def local_source(cfg):
    from bs4 import BeautifulSoup
    directory = path(cfg["local_dir"])
    if not directory.exists():
        return
    for filename in sorted(directory.rglob("*")):
        if not filename.is_file() or filename.suffix.lower() not in {".txt", ".html", ".htm", ".epub", ".pdf"}:
            continue
        try:
            if filename.stat().st_size > cfg["max_local_bytes"]:
                raise ValueError("File exceeds max_local_bytes")
            suffix = filename.suffix.lower()
            if suffix == ".epub":
                with zipfile.ZipFile(filename) as archive:
                    # Prevent compressed archives expanding beyond the configured limit.
                    if sum(x.file_size for x in archive.infolist()) > cfg["max_local_bytes"]:
                        raise ValueError("Expanded EPUB exceeds max_local_bytes")
                    import xml.etree.ElementTree as ET
                    container = ET.fromstring(archive.read("META-INF/container.xml"))
                    opf = next(x.attrib["full-path"] for x in container.iter() if x.tag.endswith("rootfile"))
                    package = ET.fromstring(archive.read(opf))
                    manifest = {x.attrib["id"]: x.attrib["href"] for x in package.iter() if x.tag.endswith("item")}
                    spine = [x.attrib["idref"] for x in package.iter() if x.tag.endswith("itemref")]
                    parts = [archive.read(str(Path(opf).parent.joinpath(manifest[item])).replace("\\", "/")) for item in spine]
                    text = "\n".join(BeautifulSoup(p, "html.parser").get_text(" ") for p in parts)
            elif suffix == ".pdf":
                from pypdf import PdfReader
                text = "\n".join(p.extract_text() or "" for p in PdfReader(filename).pages)
            else:
                text = filename.read_text(encoding="utf-8-sig", errors="strict")
            if len(text.encode("utf-8")) > cfg["max_local_bytes"]:
                raise ValueError("Extracted text exceeds max_local_bytes")
            yield {"raw": {"text": text}, "source": "local:" + filename.relative_to(directory).as_posix(),
                   "license": cfg["local_license"], "group": cfg["local_group"],
                   "document_id": digest(filename.relative_to(directory).as_posix() + text)}
        except Exception as e:
            LOG.warning("Cannot import %s: %s", filename, e)


def download(cfg, limit=None):
    report = {}
    for original in cfg["sources"]:
        spec = dict(original)
        if limit is not None:
            spec["limit"] = min(spec["limit"], limit)
        if spec["limit"] <= 0 or spec["max_bytes"] <= 0:
            raise ValueError("Every source needs positive limit and max_bytes")
        try:
            metadata, records = hf_source(spec) if "repo" in spec else web_source(spec)
            count = 0
            def counted():
                nonlocal count
                for record in records:
                    count += 1
                    yield record
            write_jsonl(f"{cfg['raw_dir']}/{spec['id']}.jsonl", counted())
            report[spec["id"]] = {"status": "ok", "rows": count, **metadata}
            LOG.info("%s: %d records", spec["id"], count)
        except Exception as e:
            # Never leave an older successful file looking like this run's output.
            path(f"{cfg['raw_dir']}/{spec['id']}.jsonl").unlink(missing_ok=True)
            report[spec["id"]] = {"status": "unavailable", "error": f"{type(e).__name__}: {e}"}
            LOG.error("%s unavailable: %s", spec["id"], e)
    write_jsonl(f"{cfg['raw_dir']}/local.jsonl", local_source(cfg))
    write_json("outputs/download_report.json", report)
    return report


def text_from_raw(raw, text_column=None):
    if text_column:
        value = raw.get(text_column)
        if not isinstance(value, str):
            raise ValueError(f"Configured text column {text_column!r} is not a string")
        return value
    for key in ("text", "content", "body", "article"):
        if isinstance(raw.get(key), str):
            return raw[key]
    raise ValueError(f"Unrecognized text schema {list(raw)}; set text_column in data.yaml")


def messages_from_raw(raw):
    messages = raw.get("messages") or raw.get("conversations")
    if isinstance(messages, list):
        roles = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant", "system": "system"}
        return [{"role": roles.get(m.get("role", m.get("from")), "invalid"),
                 "content": m.get("content", m.get("value", ""))} for m in messages]
    if isinstance(raw.get("instruction"), str) and isinstance(raw.get("output"), str):
        return [{"role": "user", "content": raw["instruction"] + "\n" + (raw.get("input") or "")},
                {"role": "assistant", "content": raw["output"]}]
    raise ValueError(f"Unknown dialogue schema: {list(raw)}")
