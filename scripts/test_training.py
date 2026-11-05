"""One real Qwen3.5-0.8B SFT optimizer step on the available CPU/GPU."""
import argparse
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["PYTHONUTF8"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_XET"] = "1"


class TeeStream:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile
    def write(self, value):
        self.logfile.write(value)
        self.logfile.flush()
        return self.terminal.write(value)
    def flush(self):
        self.logfile.flush()
        self.terminal.flush()
    def __getattr__(self, name):
        return getattr(self.terminal, name)


def main():
    from src.common import config, path, read_jsonl, write_json, write_jsonl
    from src.persona import SYSTEM, with_system_prompt
    from src.training.engine import environment, select_device, train
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="outputs/test_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--cpu-threads", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    args = parser.parse_args()
    output = path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    logfile = (output / "training.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, logfile)
    sys.stderr = TeeStream(sys.stderr, logfile)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    result = {"status": "running", "model": "Qwen/Qwen3.5-0.8B-Base", "output": str(output)}
    write_json(output / "status.json", result)
    try:
        env = environment()
        device = select_device(args.device, env["cuda_available"])
        print(f"REAL MODEL TEST | model={result['model']} | device={device} | CPU threads={args.cpu_threads}", flush=True)
        print("System prompt: " + SYSTEM, flush=True)
        print("One SFT step directly from Base for software verification; production SFT still uses merged CPT.", flush=True)
        print("Downloading/loading weights can take several minutes. Loss and checkpoint progress follow.", flush=True)
        for split in ("train", "validation"):
            rows = sorted(read_jsonl(f"data/processed/sft_{split}.jsonl"), key=lambda row: sum(len(m["content"]) for m in row["messages"]))
            if not rows:
                raise ValueError(f"Empty {split} split")
            write_jsonl(output / f"{split}.jsonl", [rows[0] | {"messages": with_system_prompt(rows[0]["messages"])}])
        cfg = config("configs/sft.yaml")
        base = config("configs/cpt.yaml")
        cfg.update(model=base["model"], tokenizer=base["tokenizer"], revision=base["revision"],
                   device=args.device, cpu_threads=args.cpu_threads, max_length=256,
                   output_dir=str(output / "adapter"), train_file=str(output / "train.jsonl"),
                   validation_file=str(output / "validation.jsonl"), test_run=True)
        cfg["training"].update(max_steps=1, gradient_accumulation_steps=1, warmup_ratio=0,
                               logging_steps=1, eval_steps=1, save_steps=1, report_to="none")
        write_json(output / "config.json", cfg)
        adapter = train("sft", cfg)
        result.update(status="passed", device=device, adapter=str(adapter), optimizer_steps=1)
        print(f"\nTEST PASSED. Adapter and checkpoint: {adapter}", flush=True)
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        raise
    finally:
        write_json(output / "status.json", result)


if __name__ == "__main__":
    main()
