import json
import pytest
from src.common import read_jsonl, write_jsonl
from src.data.prepare import clean_text, normalize, split_rows, mix_rows, validate_messages, conversation_key
from src.data.ingest import text_from_raw, messages_from_raw
from src.data.demo import EXAMPLES, EVAL


def test_cleaning_preserves_historical_ukrainian():
    text = clean_text('<p>Возлюблений, їж хлїб: ґанок є!</p><script>evil()</script> https://example.com')
    assert text == 'Возлюблений, їж хлїб: ґанок є!'
    assert normalize('  ЇЖ  ХЛІБ ') == normalize('їж хліб')


def test_jsonl_atomic_failure(tmp_path):
    target = tmp_path / 'data.jsonl'
    write_jsonl(target, [{'x': 'ї'}])
    def failing():
        yield {'x': 'bad'}
        raise ValueError('interrupted')
    with pytest.raises(ValueError):
        write_jsonl(target, failing())
    assert list(read_jsonl(target)) == [{'x': 'ї'}]


def test_disjoint_reproducible_splits():
    rows = [{'document_id': str(i), 'split_group': str(i//2)} for i in range(40)]
    a = split_rows(rows)
    assert a == split_rows(rows)
    keys = [{r['split_group'] for r in v} for v in a.values()]
    assert not keys[0] & keys[1] and not keys[0] & keys[2] and not keys[1] & keys[2]


def test_mix_without_duplicate_oversampling():
    rows = [{'group': 'general', 'n': i} for i in range(90)] + [{'group':'religious','n':i} for i in range(10)]
    assert len(mix_rows(rows, {'general': .9, 'religious': .1}, 'group')) == 100
    with pytest.raises(ValueError, match='Missing'):
        mix_rows(rows[:90], {'general': .9, 'religious': .1}, 'group')


@pytest.mark.parametrize('messages', [[], [{'role':'assistant','content':'hello'}],
    [{'role':'user','content':'a'}, {'role':'user','content':'b'}]])
def test_bad_conversations(messages):
    with pytest.raises(ValueError):
        validate_messages(messages)


def test_empty_messages_removed_before_alternation():
    result = validate_messages([{'role':'system','content':' '}, {'role':'user','content':'q'}, {'role':'assistant','content':'a'}])
    assert len(result) == 2


def test_schema_adapters():
    assert text_from_raw({'text':'текст', 'length':5}) == 'текст'
    with pytest.raises(ValueError):
        text_from_raw({'id':1})
    m = messages_from_raw({'conversations':[{'from':'human','value':'q'}, {'from':'gpt','value':'a'}]})
    assert validate_messages(m)[-1]['role'] == 'assistant'


def test_demo_size_uniqueness_and_eval_isolation():
    prompts = []
    for block in EXAMPLES.values():
        pairs = [line.split('|', 1) for line in block.strip().splitlines()]
        assert len(pairs) == 15
        assert all(len(answer) > 30 for _, answer in pairs)
        prompts.extend(question for question, _ in pairs)
    assert len(prompts) == len(set(prompts)) == 120
    assert not set(prompts) & {q for _, q, _ in EVAL}
