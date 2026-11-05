"""Numbered entry points share the same tested command implementation."""
import argparse
import json
import logging
from src.common import config, path, read_jsonl, write_json, write_jsonl


def main(command):
    parser = argparse.ArgumentParser(description=f"Orthodox Qwen UA: {command}")
    default_config = "configs/sft.yaml" if command in {"train_sft", "evaluate"} else "configs/cpt.yaml" if command in {"train_cpt", "merge"} else "configs/data.yaml"
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--create-demo", action="store_true")
    parser.add_argument("--import-dialogues", action="store_true")
    parser.add_argument("--adapter")
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output")
    parser.add_argument("--responses")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    cfg = config(args.config)
    if args.device:
        cfg["device"] = args.device
    try:
        if command == "download":
            from src.data.ingest import download
            result = download(cfg, args.limit)
            if not any(r.get("rows", 0) for r in result.values()):
                raise RuntimeError("No remote records downloaded; inspect outputs/download_report.json; local files were still imported")
        elif command == "inspect":
            result = {}
            for filename in path(cfg["raw_dir"]).glob("*.jsonl"):
                records = list(read_jsonl(filename))
                result[filename.name] = {"rows": len(records), "columns": sorted({k for r in records for k in r["raw"]}),
                                         "first_record": records[0] if records else None}
            write_json("outputs/inspection.json", result)
        elif command == "clean":
            from src.data.prepare import clean_corpus
            clean_corpus(cfg)
        elif command == "prepare_cpt":
            from src.data.prepare import prepare_cpt
            tokenizer = None
            if not args.skip_tokenizer:
                from src.training.tokenization import load_tokenizer
                tokenizer = load_tokenizer(cfg["model"], cfg["model_revision"])
            prepare_cpt(cfg, tokenizer)
        elif command == "prepare_sft":
            from src.data.demo import create_demo, create_eval
            from src.data.prepare import prepare_sft
            if args.create_demo:
                create_demo()
            if not path("data/eval/prompts.jsonl").exists():
                create_eval()
            if args.import_dialogues:
                import_dialogues(cfg)
            prepare_sft(cfg)
        elif command == "validate":
            from src.data.prepare import validate_files
            print(json.dumps(validate_files(cfg), indent=2))
        elif command in {"train_cpt", "train_sft"}:
            from src.training.engine import preflight, train
            if args.preflight:
                print(json.dumps(preflight(cfg), indent=2))
            else:
                train("sft" if command == "train_sft" else "cpt", cfg, args.resume, args.smoke)
        elif command == "merge":
            from src.training.engine import merge
            print(merge(cfg, args.adapter, args.output))
        elif command == "evaluate":
            from src.evaluation.run import evaluate
            model_id = args.model or cfg["merged_dir"]
            revision = args.revision
            if model_id == "Qwen/Qwen3.5-0.8B-Base" and revision is None:
                revision = config("configs/data.yaml")["model_revision"]
            evaluate(model_id, revision=revision, responses=args.responses,
                     device_requested=cfg.get("device", "auto"),
                     output=args.output or "outputs/evaluation", limit=args.limit)
    except Exception as e:
        logging.error("%s failed: %s: %s", command, type(e).__name__, e)
        raise SystemExit(1) from e


def import_dialogues(cfg):
    from src.data.ingest import messages_from_raw
    from src.data.prepare import validate_messages
    imported = []
    for spec in cfg["sources"]:
        if spec["group"] != "dialogue":
            continue
        filename = path(f"{cfg['raw_dir']}/{spec['id']}.jsonl")
        if not filename.exists():
            continue
        for row in read_jsonl(filename):
            try:
                messages = validate_messages(messages_from_raw(row["raw"]))
                imported.append({k:v for k,v in row.items() if k != "raw"} |
                                {"messages": messages, "synthetic": False,
                                 "category": "everyday_conversation", "split_group": row["document_id"]})
            except ValueError as e:
                logging.warning("Rejected source dialogue %s: %s", row["document_id"], e)
    write_jsonl("data/sft_sources/imported.jsonl", imported)
