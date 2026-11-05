"""Offline end-to-end CPU smoke with a tiny randomly initialized Qwen3.5, never 0.8B."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/smoke")
    args = parser.parse_args()
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from src.common import path, write_json, write_jsonl
    from src.training.engine import train, merge
    from src.evaluation.run import evaluate
    from src.persona import with_system_prompt
    torch.set_num_threads(2)
    torch.manual_seed(3407)
    directory = path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "base/config.json").exists():
        raise RuntimeError("outputs/smoke already exists. Use a fresh output directory for a repeat run")
    words = ["[PAD]", "[UNK]", "[EOS]", "system", "user", "assistant", ":", ";", "привіт", "вітаю", "українська", "мова"]
    backend = Tokenizer(models.WordLevel({word:i for i,word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]")
    # Test fixture only. Production always uses the verified official Qwen template.
    tokenizer.chat_template = "{% for m in messages %}{{ m.role + ': ' + m.content + ' ; ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    tokenizer.save_pretrained(directory / "base")
    cfg = Qwen3_5TextConfig(vocab_size=len(words), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2,
        linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        tie_word_embeddings=True, eos_token_id=2, pad_token_id=0,
        rope_parameters={"rope_type":"default", "rope_theta":10000, "partial_rotary_factor":1.0})
    Qwen3_5ForCausalLM(cfg).save_pretrained(directory / "base")
    cpt = [{"text":"привіт українська мова", "document_id":str(i), "source":"smoke", "license":"CC0-1.0"} for i in range(4)]
    sft = [{"messages":with_system_prompt([{"role":"user","content":"привіт"},{"role":"assistant","content":"вітаю"}]),
            "document_id":str(i), "source":"smoke", "license":"CC0-1.0"} for i in range(4)]
    write_jsonl(directory / "cpt.jsonl", cpt)
    write_jsonl(directory / "sft.jsonl", sft)
    common = {"seed":3407, "max_length":256, "device":"auto", "prefer_bf16":True,
              "min_free_vram_gib":0, "prefer_8bit_optimizer":False,
              "lora":{"r":2,"lora_alpha":4,"lora_dropout":0.,"target_modules":["q_proj","v_proj","up_proj","down_proj"]},
              "training":{"per_device_train_batch_size":1,"per_device_eval_batch_size":1,
                          "gradient_accumulation_steps":1,"learning_rate":.001,"max_steps":1,
                          "warmup_ratio":0,"gradient_checkpointing":True,"logging_steps":1,
                          "save_steps":1,"eval_steps":1,"eval_strategy":"steps","report_to":"none"}}
    cpt_cfg = common | {"model":str(directory/"base"), "tokenizer":str(directory/"base"),
                       "train_file":str(directory/"cpt.jsonl"),"validation_file":str(directory/"cpt.jsonl"),
                       "output_dir":str(directory/"cpt_adapter"),"merged_dir":str(directory/"cpt_merged")}
    train("cpt", cpt_cfg)
    cpt_cfg["training"] = dict(cpt_cfg["training"], max_steps=2)
    train("cpt", cpt_cfg, resume=str(directory/"cpt_adapter/checkpoint-1"))
    merge(cpt_cfg)
    sft_cfg = common | {"model":str(directory/"cpt_merged"),"tokenizer":str(directory/"cpt_merged"),
                       "train_file":str(directory/"sft.jsonl"),"validation_file":str(directory/"sft.jsonl"),
                       "output_dir":str(directory/"sft_adapter"),"merged_dir":str(directory/"final_merged"), "packing":False}
    train("sft", sft_cfg)
    merge(sft_cfg)
    write_jsonl(directory / "prompts.jsonl", [{"document_id":"eval","category":"smoke",
                "messages":with_system_prompt([{"role":"user","content":"привіт"}])}])
    evaluate(str(directory/"final_merged"), prompts=str(directory/"prompts.jsonl"),
             output=str(directory/"evaluation"), max_new_tokens=4)
    write_json(directory / "result.json", {"status":"passed", "model":"random tiny Qwen3.5",
        "checks":["CPT backward", "checkpoint resume", "CPT merge", "SFTTrainer assistant labels", "SFT merge", "generation evaluation"],
        "warning":"This validates software integration, not 0.8B quality, BF16 or CUDA memory."})
    print("Tiny CPU CPT -> resume -> merge -> SFT -> merge -> evaluation: passed")


if __name__ == "__main__":
    main()
