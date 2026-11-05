import pytest
from src.training.tokenization import MaskedCollator, encode_sft, encode_cpt, chat_stop_ids


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    chat_template = 'test fixture'
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        result = ''.join(m['role'] + ':' + m['content'] + ';' for m in messages)
        return result + ('assistant:' if add_generation_prompt else '')
    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(c) for c in text]}


def test_multiturn_assistant_mask_and_padding():
    tok = TinyTokenizer()
    messages = [{'role':'system','content':'s'}, {'role':'user','content':'q'}, {'role':'assistant','content':'a'},
                {'role':'user','content':'r'}, {'role':'assistant','content':'b'}]
    item = encode_sft({'messages':messages}, tok, 1000)
    trained = ''.join(chr(label) for label in item['labels'] if label != -100)
    assert trained == 'a;b;'
    shorter = encode_sft({'messages':messages[:3]}, tok, 1000)
    batch = MaskedCollator(tok)([item, shorter])
    assert batch['labels'][1, -1].item() == -100
    assert batch['attention_mask'][1, -1].item() == 0
    with pytest.raises(ValueError, match='max_length'):
        encode_sft({'messages':messages}, tok, 3)


def test_cpt_eos_and_tail():
    items = encode_cpt([{'text':'abcd'}], TinyTokenizer(), 3)
    assert [i for row in items for i in row['input_ids']] == [97,98,99,100,1]


def test_chat_terminator_is_derived_separately_from_document_eos():
    class WithSpecials(TinyTokenizer):
        all_special_ids = [1, ord(';')]
    assert chat_stop_ids(WithSpecials()) == [1, ord(';')]


@pytest.mark.parametrize('requested,available,expected', [('auto',False,'cpu'),('auto',True,'cuda'),('cpu',True,'cpu'),('cuda',True,'cuda')])
def test_device_selection(requested, available, expected):
    from src.training.engine import select_device
    assert select_device(requested, available) == expected


def test_unavailable_gpu_is_explicit_error():
    from src.training.engine import select_device
    with pytest.raises(RuntimeError, match='CUDA requested'):
        select_device('cuda', False)


def test_sft_uses_canonical_prompt():
    from src.persona import SYSTEM, with_system_prompt
    messages = [{'role':'system','content':'old'}, {'role':'user','content':'q'}, {'role':'assistant','content':'a'}]
    canonical = with_system_prompt(messages)
    assert canonical[0] == {'role':'system','content':SYSTEM}
    assert with_system_prompt(canonical) == canonical
    assert 'сервітор православний проповідник' in SYSTEM


def tiny_config(vocab_size=128):
    from transformers import Qwen3_5TextConfig
    return Qwen3_5TextConfig(vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        layer_types=['linear_attention','full_attention'], linear_num_key_heads=2,
        linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        max_position_embeddings=2048, tie_word_embeddings=True,
        rope_parameters={'rope_type':'default','rope_theta':10000,'partial_rotary_factor':1.0})


def test_qwen35_lora_backward_and_merge(tmp_path):
    import torch
    from transformers import Qwen3_5ForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.set_num_threads(2)
    torch.manual_seed(3407)
    model = Qwen3_5ForCausalLM(tiny_config())
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, task_type='CAUSAL_LM',
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    ids = torch.tensor([[3,4,5,6,7]])
    labels = torch.tensor([[-100,-100,-100,6,7]])
    loss = model(input_ids=ids, labels=labels, use_cache=False).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in model.named_parameters() if 'lora_B' in n)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.01)
    opt.step()
    model.eval()
    with torch.no_grad():
        before = model(input_ids=ids).logits
        merged = model.merge_and_unload(safe_merge=True)
        after = merged(input_ids=ids).logits
    assert torch.allclose(before, after, atol=1e-5)
    merged.save_pretrained(tmp_path)
    from src.training.engine import load_text_model
    loaded = load_text_model(str(tmp_path), dtype=torch.float32)
    with torch.no_grad():
        assert torch.allclose(loaded(input_ids=ids).logits, after, atol=1e-5)


def test_official_vlm_to_text_loading(tmp_path):
    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from src.training.engine import load_text_model
    cfg = Qwen3_5Config(text_config=tiny_config().to_dict(), vision_config={
        'hidden_size':32, 'intermediate_size':64, 'depth':1, 'num_heads':4,
        'out_hidden_size':32, 'patch_size':2, 'spatial_merge_size':1,
        'temporal_patch_size':1, 'num_position_embeddings':16})
    original = Qwen3_5ForConditionalGeneration(cfg)
    original.save_pretrained(tmp_path)
    loaded = load_text_model(str(tmp_path), dtype=torch.float32)
    assert torch.equal(loaded.model.embed_tokens.weight, original.model.language_model.embed_tokens.weight)
