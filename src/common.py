"""Portable configuration, strict JSONL and atomic outputs."""
import hashlib
import json
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("orthodox_qwen")


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def config(filename):
    import yaml
    with path(filename).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(filename):
    with path(filename).open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected JSON object")
                    yield row
                except (ValueError, TypeError) as e:
                    raise ValueError(f"{filename}:{number}: {e}") from e


def write_jsonl(filename, rows):
    target = path(filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(filename, data):
    target = path(filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, target)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
