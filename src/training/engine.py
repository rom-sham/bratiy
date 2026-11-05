"""CPT Trainer and SFTTrainer share the same audited tokenization/collation."""
import importlib.metadata
import json
import logging
import math
import platform
import sys

from src.common import path, read_jsonl, write_json
from src.training.tokenization import MaskedCollator, encode_cpt, encode_sft, load_tokenizer

LOG = logging.getLogger(__name__)


def environment():
    import torch
    versions = {}
    for package in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "bitsandbytes"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {"python": sys.version, "platform": platform.platform(), "packages": versions,
              "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        result.update(gpu=torch.cuda.get_device_name(), free_vram_gib=torch.cuda.mem_get_info()[0]/2**30,
                      total_vram_gib=torch.cuda.mem_get_info()[1]/2**30, bf16=torch.cuda.is_bf16_supported())
    return result


def select_device(requested, cuda_available):
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA requested but unavailable; select device: auto or cpu")
    return "cuda" if requested == "cuda" or (requested == "auto" and cuda_available) else "cpu"


def preflight(cfg):
    env = environment()
    env["selected_device"] = select_device(cfg.get("device", "auto"), env["cuda_available"])
    write_json("outputs/environment.json", env)
    LOG.info("Device: %s | CPU: %s | CUDA build: %s | GPU: %s", env["selected_device"],
             platform.processor() or platform.machine(), env["cuda_build"], env.get("gpu", "unavailable"))
    if env["selected_device"] == "cpu" and "cpu_threads" in cfg:
        import torch
        torch.set_num_threads(int(cfg["cpu_threads"]))
    if env["selected_device"] == "cuda" and env["free_vram_gib"] < cfg["min_free_vram_gib"]:
        raise RuntimeError(f"Free VRAM {env['free_vram_gib']:.2f} GiB < configured minimum; lower max_length or free GPU memory")
    for key in ("train_file", "validation_file"):
        if not path(cfg[key]).is_file():
            raise FileNotFoundError(f"Missing {cfg[key]}; prepare and validate data first")
    return env


def resolve_model(value):
    candidate = path(value)
    if candidate.exists() or value.startswith(("outputs/", "data/", "./", "../")):
        if not candidate.exists():
            raise FileNotFoundError(f"Local checkpoint missing: {candidate}. Merge CPT before SFT")
        return str(candidate)
    return value


def load_text_model(model_id, revision=None, dtype=None):
    from transformers import AutoConfig, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    cfg = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    commit = getattr(cfg, "_commit_hash", None)
    if cfg.model_type == "qwen3_5":
        # The official text class remaps model.language_model.* keys in the VLM checkpoint.
        cfg = cfg.text_config
        cfg._commit_hash = commit
    if not isinstance(cfg, Qwen3_5TextConfig):
        raise ValueError(f"Expected Qwen3.5 text config, got {type(cfg).__name__}")
    model, loading = Qwen3_5ForCausalLM.from_pretrained(model_id, revision=revision, config=cfg,
                              dtype=dtype, attn_implementation="eager", output_loading_info=True)
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise RuntimeError(f"Incomplete text checkpoint load: {loading}")
    return model


def optimizer_name(cfg, env):
    if cfg.get("prefer_8bit_optimizer") and env["selected_device"] == "cuda":
        try:
            import bitsandbytes as bnb
            import torch
            # Check the native backend actually performs an update on this device.
            parameter = torch.nn.Parameter(torch.zeros(4096, device="cuda"))
            opt = bnb.optim.AdamW8bit([parameter])
            parameter.sum().backward()
            opt.step()
            del opt, parameter
            return "adamw_bnb_8bit"
        except Exception as e:
            LOG.warning("AdamW 8-bit unavailable; using adamw_torch: %s", e)
    return "adamw_torch"


def train(stage, cfg, resume=None, smoke=False):
    import torch
    try:
        return _train(stage, cfg, resume, smoke)
    except torch.OutOfMemoryError as e:
        output = path(cfg["output_dir"] + ("_smoke" if smoke else ""))
        write_json(output / "oom.json", {"error": str(e), "suggestion": "Reduce max_length, free VRAM, resume last completed checkpoint"})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError("OOM during model setup or training; see oom.json") from e


def _train(stage, cfg, resume=None, smoke=False):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    env = preflight(cfg)
    set_seed(cfg["seed"])
    model_id = resolve_model(cfg["model"])
    tokenizer = load_tokenizer(resolve_model(cfg.get("tokenizer", cfg["model"])), cfg.get("revision"))
    use_bf16 = bool(cfg.get("prefer_bf16") and env["selected_device"] == "cuda" and env.get("bf16"))
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = load_text_model(model_id, cfg.get("revision"), dtype)
    model.config.use_cache = False
    targets = cfg["lora"]["target_modules"]
    found = {name.rsplit(".", 1)[-1] for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    if set(targets) - found:
        raise ValueError(f"LoRA targets absent from model: {set(targets)-found}")
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **cfg["lora"]))
    if cfg["training"].get("gradient_checkpointing"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    prepared = {}
    for split, key in (("train", "train_file"), ("validation", "validation_file")):
        rows = list(read_jsonl(cfg[key]))
        if smoke:
            rows = rows[:4]
        features = ([encode_sft(r, tokenizer, cfg["max_length"]) for r in rows] if stage == "sft"
                    else encode_cpt(rows, tokenizer, cfg["max_length"]))
        if not features:
            raise ValueError(f"No trainable records in {split}")
        prepared[split] = Dataset.from_list(features)
    options = dict(cfg["training"])
    output = path(cfg["output_dir"] + ("_smoke" if smoke else ""))
    options.update(output_dir=str(output), seed=cfg["seed"], bf16=use_bf16, fp16=False,
                   use_cpu=env["selected_device"] == "cpu", dataloader_pin_memory=env["selected_device"] == "cuda",
                   optim=optimizer_name(cfg, env), remove_unused_columns=False,
                   gradient_checkpointing_kwargs={"use_reentrant": False})
    if smoke:
        options.update(max_steps=1, gradient_accumulation_steps=1, logging_steps=1, save_steps=1,
                       eval_steps=1, report_to="none", warmup_ratio=0)
    # Transformers v5.22 removed warmup_ratio. Convert YAML ratio to optimizer steps.
    ratio = options.pop("warmup_ratio", 0)
    steps = options.get("max_steps", -1)
    if steps <= 0:
        batches = math.ceil(len(prepared["train"]) / options["per_device_train_batch_size"])
        steps = math.ceil(batches / options["gradient_accumulation_steps"]) * options.get("num_train_epochs", 1)
    options["warmup_steps"] = math.ceil(steps * ratio)
    class MemoryCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if env["selected_device"] == "cuda":
                peak = torch.cuda.max_memory_allocated()/2**30
                LOG.info("step=%s CUDA peak allocated=%.3f GiB", state.global_step, peak)

    if stage == "sft":
        from trl import SFTConfig, SFTTrainer
        if cfg.get("packing"):
            raise ValueError("Packing is disabled for explicit assistant labels; use packing: false. Native packing requires a separately validated template/mask path")
        args = SFTConfig(**options, max_length=cfg["max_length"], packing=False,
                         dataset_kwargs={"skip_prepare_dataset": True}, assistant_only_loss=False)
        trainer = SFTTrainer(model=model, args=args, processing_class=tokenizer,
                             train_dataset=prepared["train"], eval_dataset=prepared["validation"],
                             data_collator=MaskedCollator(tokenizer), callbacks=[MemoryCallback()])
        if trainer.train_dataset[0]["labels"] != prepared["train"][0]["labels"]:
            raise RuntimeError("SFTTrainer modified precomputed assistant labels")
    else:
        args = TrainingArguments(**options)
        trainer = Trainer(model=model, args=args, processing_class=tokenizer,
                          train_dataset=prepared["train"], eval_dataset=prepared["validation"],
                          data_collator=MaskedCollator(tokenizer), callbacks=[MemoryCallback()])
    # Capture exact resolved base, version set, data and config for merge/reproduction.
    write_json(output / "run_manifest.json", {"stage": stage, "config": cfg, "environment": env,
               "base_model": model_id, "base_revision": getattr(model.config, "_commit_hash", None) or cfg.get("revision"),
               "assistant_labels": stage == "sft", "smoke": smoke})
    try:
        trainer.train(resume_from_checkpoint=str(path(resume)) if resume else None)
        trainer.save_model(str(output))
        tokenizer.save_pretrained(output)
        trainer.save_state()
        write_json(output / "metrics.json", trainer.evaluate())
    except torch.OutOfMemoryError as e:
        write_json(output / "oom.json", {"error": str(e), "suggestion": "Reduce max_length, free VRAM; resume last completed checkpoint", "environment": env})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError("Out of memory; see oom.json. Last completed checkpoint is safe; reduce sequence length and resume") from e
    return output


def merge(cfg, adapter=None, output=None):
    import torch
    from peft import PeftModel
    adapter_path = path(adapter or cfg["output_dir"])
    destination = path(output or cfg["merged_dir"])
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Merge destination is not empty: {destination}")
    manifest_path = adapter_path / "run_manifest.json"
    if not manifest_path.exists():
        raise ValueError("Missing run_manifest.json; cannot verify adapter base revision")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = resolve_model(manifest["base_model"])
    model = load_text_model(base, manifest.get("base_revision"), torch.float32)
    merged = PeftModel.from_pretrained(model, adapter_path).merge_and_unload(safe_merge=True)
    merged.save_pretrained(destination, safe_serialization=True)
    load_tokenizer(str(adapter_path)).save_pretrained(destination)
    write_json(destination / "merge_manifest.json", manifest | {"adapter": str(adapter_path)})
    return destination
