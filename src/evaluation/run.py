"""Generation reports; diagnostics are explicitly not subjective quality scores."""
import re
from collections import Counter
from src.common import path, read_jsonl, write_json
from src.persona import with_system_prompt

RUBRIC = {
    "ukrainian": "0–4: граматика, природність, відсутність небажаного перемикання мови",
    "archaic_style": "0–4: доречна, зрозуміла архаїка без перевантаження",
    "persona": "0–4: послідовність Братія та чесне позначення вигаданого образу",
    "humor": "0–4: доброзичлива іронія, доречний жарт",
    "ordinary": "0–4: пряма корисна відповідь без зайвої проповіді",
    "terminology": "0–4: точність православних термінів; перевірити джерела вручну",
    "warhammer": "0–4: зв'язність оригінальної сцени; відокремлення фантастики й реальної віри",
    "general": "0–4: збереження здібностей; порівняти ту саму відповідь базової моделі",
}


def diagnostics(text, reference=None):
    words = re.findall(r"\w+", text.casefold())
    trigrams = list(zip(words, words[1:], words[2:]))
    return {"words": len(words), "repeated_trigram_fraction":
            (1-len(set(trigrams))/len(trigrams)) if trigrams else 0,
            "exact_reference_match": text.strip().casefold() == reference.strip().casefold() if reference else None}


def save_report(results, output, model=None):
    report = {"model": model, "rubric": RUBRIC, "results": results,
              "note": "Automatic metrics are diagnostics. Human ratings are intentionally null until reviewed."}
    write_json(str(output) + ".json", report)
    lines = ["# Orthodox Qwen UA — evaluation", "", f"Model: {model}", "",
             "Автоматичні метрики — діагностика. Стиль, гумор і факти оцінює людина за шкалою 0–4.", ""]
    for key, value in RUBRIC.items():
        lines.append(f"- {key}: {value}")
    for row in results:
        lines.extend(["", f"## {row['category']}", "", row["prompt"], "", row["response"], "",
                      f"Diagnostics: {row['diagnostics']}", "", "Human score: ___ / 4"])
    target = path(str(output) + ".md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return report


def evaluate(model_id=None, revision=None, prompts="data/eval/prompts.jsonl", output="outputs/evaluation",
             responses=None, max_new_tokens=192, limit=None, device_requested="auto"):
    import torch
    rows = list(read_jsonl(prompts))
    rows = [row | {"messages": with_system_prompt(row["messages"])} for row in rows]
    if limit:
        rows = rows[:limit]
    if responses:
        generated = {r["document_id"]: r["response"] for r in read_jsonl(responses)}
    else:
        from src.training.engine import load_text_model, resolve_model, select_device
        from src.training.tokenization import load_tokenizer, render, chat_stop_ids
        model_id = resolve_model(model_id)
        tokenizer = load_tokenizer(model_id, revision)
        device = select_device(device_requested, torch.cuda.is_available())
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        model = load_text_model(model_id, revision, dtype).to(device).eval()
        generated = {}
        for row in rows:
            inputs = tokenizer(render(tokenizer, row["messages"], True), add_special_tokens=False, return_tensors="pt").to(device)
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id, eos_token_id=chat_stop_ids(tokenizer))
            generated[row["document_id"]] = tokenizer.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    results = []
    for row in rows:
        if row["document_id"] not in generated:
            raise ValueError(f"Missing evaluation response {row['document_id']}")
        response = generated[row["document_id"]]
        results.append({"document_id": row["document_id"], "category": row["category"],
                        "prompt": row["messages"][-1]["content"], "response": response,
                        "diagnostics": diagnostics(response, row.get("reference")), "human_score": None})
    return save_report(results, output, model_id)
