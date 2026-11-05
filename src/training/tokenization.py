"""Template-native serialization and explicit completion masks, without token literals."""
from src.data.prepare import validate_messages
from src.persona import with_system_prompt


def load_tokenizer(model, revision=None):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer lacks both pad and EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def render(tokenizer, messages, generation=False):
    if not tokenizer.chat_template:
        raise ValueError("No official chat template; configure a reviewed compatible template")
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=generation, enable_thinking=False)


def encode_sft(row, tokenizer, max_length):
    messages = with_system_prompt(validate_messages(row["messages"]))
    full_text = render(tokenizer, messages)
    ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(ids) > max_length:
        # Do not silently truncate away answers or mask boundaries.
        raise ValueError(f"SFT dialogue has {len(ids)} tokens, max_length={max_length}; shorten or filter it")
    labels = [-100] * len(ids)
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = tokenizer(render(tokenizer, messages[:i], generation=True), add_special_tokens=False)["input_ids"]
        end = tokenizer(render(tokenizer, messages[:i+1]), add_special_tokens=False)["input_ids"]
        # Some templates rewrite earlier assistant turns. Fail closed rather than train wrong spans.
        if end[:len(prefix)] != prefix or ids[:len(end)] != end:
            raise ValueError("Official template is not prefix-preserving for this conversation; split into reviewed single turns")
        labels[len(prefix):len(end)] = ids[len(prefix):len(end)]
    if not any(x != -100 for x in labels):
        raise ValueError("No assistant tokens to train")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def encode_cpt(rows, tokenizer, max_length):
    if tokenizer.eos_token_id is None:
        raise ValueError("CPT requires an EOS token")
    result = []
    for row in rows:
        ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        # Keep document boundaries. No train/validation concatenation or dropped remainder.
        for start in range(0, len(ids), max_length):
            block = ids[start:start+max_length]
            if len(block) >= 2:
                result.append({"input_ids": block, "attention_mask": [1]*len(block), "labels": block.copy()})
    return result


def chat_stop_ids(tokenizer):
    """Derive the turn terminator from the official template, without token literals."""
    sample = [{"role": "user", "content": "Привіт"}, {"role": "assistant", "content": "Вітаю"}]
    prefix = tokenizer(render(tokenizer, sample[:1], True), add_special_tokens=False)["input_ids"]
    complete = tokenizer(render(tokenizer, sample), add_special_tokens=False)["input_ids"]
    if complete[:len(prefix)] != prefix:
        raise ValueError("Template is not prefix-preserving for generation")
    specials = set(tokenizer.all_special_ids)
    closing = next((token for token in reversed(complete[len(prefix):]) if token in specials), None)
    if closing is None:
        closing = tokenizer.eos_token_id
    return sorted({token for token in (tokenizer.eos_token_id, closing) if token is not None})


class MaskedCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        import torch
        length = max(len(f["input_ids"]) for f in features)
        batch = {}
        for key, fill in (("input_ids", self.pad_id), ("attention_mask", 0), ("labels", -100)):
            batch[key] = torch.tensor([list(f[key]) + [fill]*(length-len(f[key])) for f in features], dtype=torch.long)
        return batch
