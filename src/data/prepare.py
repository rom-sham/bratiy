"""Cleaning, leak-resistant splits and sampling without duplicate oversampling."""
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict

from src.common import digest, path, read_jsonl, write_json, write_jsonl
from src.data.ingest import text_from_raw
from src.persona import with_system_prompt


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def clean_text(text):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "nav"]):
        tag.decompose()
    text = soup.get_text(" ")
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # NFC retains Ukrainian letters and historical spelling; NFKC only for dedup keys.
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def language_info(text):
    from langdetect import DetectorFactory, detect_langs
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 3407
    try:
        predictions = detect_langs(text)
        language, confidence = predictions[0].lang, predictions[0].prob
    except LangDetectException:
        language, confidence = "unknown", 0.0
    fragments = []
    for i, sentence in enumerate(re.split(r"(?<=[.!?])\s+", text)):
        if len(sentence) < 40:
            continue
        try:
            p = detect_langs(sentence)[0]
            if p.lang == "ru" and p.prob >= 0.95:
                fragments.append({"sentence": i, "confidence": round(p.prob, 4)})
        except LangDetectException:
            pass
    return {"language": language, "language_confidence": confidence, "russian_fragments": fragments}


def clean_corpus(cfg):
    seen = set()
    counts = Counter()
    text_columns = {s.get("repo", s.get("url")): s.get("text_column") for s in cfg["sources"]}
    def records():
        for filename in sorted(path(cfg["raw_dir"]).glob("*.jsonl")):
            for row in read_jsonl(filename):
                if row["group"] == "dialogue":
                    continue
                try:
                    text = clean_text(text_from_raw(row["raw"], text_columns.get(row["source"])))
                except ValueError:
                    counts["schema_rejected"] += 1
                    continue
                if not cfg["min_chars"] <= len(text) <= cfg["max_chars"]:
                    counts["length_rejected"] += 1
                    continue
                if "\ufffd" in text or "\x00" in text or sum(ord(c) < 32 for c in text) > 0:
                    counts["corrupt_rejected"] += 1
                    continue
                # Common UTF-8 decoded as Windows-1251: flag sequences, not archaic letters.
                if len(re.findall(r"[РС][\u0400-\u04ff]", text)) > len(text) * 0.12:
                    counts["mojibake_rejected"] += 1
                    continue
                key = digest(normalize(text))
                if key in seen:
                    counts["duplicates"] += 1
                    continue
                lang = language_info(text)
                if lang["language"] not in {"uk", "unknown"} and lang["language_confidence"] >= cfg["language_confidence"]:
                    # Historical religious texts go to review rather than being destroyed.
                    if row["group"] == "religious":
                        counts["religious_language_review"] += 1
                    else:
                        counts["language_rejected"] += 1
                        continue
                seen.add(key)
                counts["kept"] += 1
                yield {k: v for k, v in row.items() if k != "raw"} | {"text": text, **lang,
                       "origin_metadata": {k:v for k,v in row["raw"].items() if k not in {"text", "content", "body", "article", text_columns.get(row["source"]) }},
                       "content_hash": key, "requires_language_review": lang["language"] != "uk"}
    write_jsonl(f"{cfg['processed_dir']}/clean.jsonl", records())
    write_json("outputs/clean_report.json", dict(counts))


def split_rows(rows, seed=3407, validation=0.1, evaluation=0.1):
    if validation <= 0 or evaluation <= 0 or validation + evaluation >= 1:
        raise ValueError("validation/evaluation fractions must be positive and sum to < 1")
    # Group all assistant turns from the same original document before splitting.
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("split_group", row["document_id"])].append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 3:
        raise ValueError("Need at least three unique document groups for disjoint splits")
    ne = max(1, int(len(keys) * evaluation))
    nv = max(1, int(len(keys) * validation))
    return {"evaluation": [r for k in keys[:ne] for r in groups[k]],
            "validation": [r for k in keys[ne:ne+nv] for r in groups[k]],
            "train": [r for k in keys[ne+nv:] for r in groups[k]]}


def mix_rows(rows, weights, key):
    if not weights or any(v < 0 for v in weights.values()) or not math.isclose(sum(weights.values()), 1.0):
        raise ValueError("Mixture weights must be nonnegative and sum to 1")
    pools = defaultdict(list)
    for row in rows:
        pools[row[key]].append(row)
    active = {k: v for k, v in weights.items() if v > 0}
    missing = set(active) - set(pools)
    if missing:
        raise ValueError(f"Missing mixture groups {sorted(missing)}; import data or explicitly change weights")
    size = int(min(len(pools[k]) / weight for k, weight in active.items()))
    chosen = [r for k, weight in active.items() for r in pools[k][:int(size * weight)]]
    if not chosen:
        raise ValueError("Mixture too small; increase source limits")
    return chosen


def prepare_cpt(cfg, tokenizer=None):
    rows = list(read_jsonl(f"{cfg['processed_dir']}/clean.jsonl"))
    # Warhammer and dialogue sources remain excluded unless weights explicitly include them.
    rows = [r for r in rows if cfg["cpt_weights"].get(r["group"], 0) > 0]
    splits = split_rows(rows, cfg["seed"], cfg["validation_fraction"], cfg["evaluation_fraction"])
    report = {}
    for name, records in splits.items():
        random.Random(cfg["seed"]).shuffle(records)
        if name == "train":
            records = mix_rows(records, cfg["cpt_weights"], "group")
        if tokenizer is not None:
            for row in records:
                row["token_count"] = len(tokenizer(row["text"], add_special_tokens=False)["input_ids"]) + 1
        write_jsonl(f"{cfg['processed_dir']}/cpt_{name}.jsonl", records)
        report[name] = {"documents": len(records), "groups": dict(Counter(r["group"] for r in records)),
                        "tokens_including_eos": sum(r["token_count"] for r in records) if tokenizer else None}
    write_json("outputs/cpt_data_report.json", report)
    return report


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    cleaned = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("Only textual messages are supported")
        if message["content"].strip():
            cleaned.append({"role": message.get("role"), "content": message["content"].strip()})
    offset = int(bool(cleaned) and cleaned[0]["role"] == "system")
    turns = cleaned[offset:]
    if len(turns) < 2 or len(turns) % 2:
        raise ValueError("Conversation must contain complete user/assistant pairs")
    for i, message in enumerate(turns):
        if message["role"] != ("user" if i % 2 == 0 else "assistant"):
            raise ValueError("Invalid user/assistant alternation")
    return cleaned


def conversation_key(messages):
    return digest(json.dumps([(m["role"], normalize(m["content"])) for m in messages if m["role"] != "system"], ensure_ascii=False))


def prepare_sft(cfg):
    rows, seen = [], set()
    eval_rows = list(read_jsonl("data/eval/prompts.jsonl"))
    reserved = {normalize(m["content"]) for row in eval_rows for m in row["messages"] if m["role"] == "user"}
    for filename in sorted(path("data/sft_sources").glob("*.jsonl")):
        for row in read_jsonl(filename):
            messages = with_system_prompt(validate_messages(row["messages"]))
            if sum(len(m["content"]) for m in messages) > cfg["max_chars"]:
                continue
            if any(normalize(m["content"]) in reserved for m in messages if m["role"] == "user"):
                raise ValueError(f"Evaluation prompt in SFT source {filename}")
            key = conversation_key(messages)
            if key in seen:
                continue
            seen.add(key)
            if not row.get("source") or not row.get("license") or row.get("category") not in cfg["sft_weights"]:
                raise ValueError(f"Missing provenance/category in {filename}")
            rows.append(row | {"messages": messages, "document_id": key})
    splits = split_rows(rows, cfg["seed"], cfg["validation_fraction"], cfg["evaluation_fraction"])
    splits["train"] = mix_rows(splits["train"], cfg["sft_weights"], "category")
    report = {}
    for name, records in splits.items():
        write_jsonl(f"{cfg['processed_dir']}/sft_{name}.jsonl", records)
        report[name] = dict(Counter(r["category"] for r in records))
    write_json("outputs/sft_data_report.json", report)
    return report


def validate_files(cfg):
    report = {}
    reserved = {normalize(m["content"]) for row in read_jsonl("data/eval/prompts.jsonl") for m in row["messages"] if m["role"] == "user"}
    for stage in ("cpt", "sft"):
        seen_ids, seen_content, seen_prompts = set(), set(), set()
        for split in ("train", "validation", "evaluation"):
            records = list(read_jsonl(f"{cfg['processed_dir']}/{stage}_{split}.jsonl"))
            if not records:
                raise ValueError(f"Empty {stage}/{split}")
            ids, content, prompts = set(), set(), set()
            for row in records:
                for key in ("source", "license", "document_id"):
                    if not row.get(key):
                        raise ValueError(f"Missing {key} in {stage}/{split}")
                ids.add(row["document_id"])
                if stage == "sft":
                    messages = validate_messages(row["messages"])
                    content.add(conversation_key(messages))
                    prompts.update(normalize(m["content"]) for m in messages if m["role"] == "user")
                else:
                    if not row.get("text") or not row.get("language"):
                        raise ValueError("Missing CPT text/language")
                    content.add(digest(normalize(row["text"])))
            if len(content) != len(records):
                raise ValueError(f"Duplicate content in {stage}/{split}")
            if ids & seen_ids or content & seen_content or prompts & seen_prompts or prompts & reserved:
                raise ValueError(f"Data leakage in {stage}/{split}")
            seen_ids.update(ids)
            seen_content.update(content)
            seen_prompts.update(prompts)
            report[f"{stage}_{split}"] = len(records)
    write_json("outputs/validation_report.json", report)
    return report
