import pytest
from src.data.ingest import download
from src.evaluation.run import diagnostics, save_report


def test_source_failure_continues(monkeypatch, tmp_path):
    import src.data.ingest as mod
    from src.common import read_jsonl
    def source(spec):
        if spec['id'] == 'bad':
            raise RuntimeError('source unavailable')
        return {'license':'unknown'}, iter([{'raw':{'text':'x'},'document_id':'1'}])
    monkeypatch.setattr(mod, 'hf_source', source)
    monkeypatch.setattr(mod, 'write_json', lambda *args: None)
    specs = [{'id':name,'repo':name,'limit':2,'max_bytes':100} for name in ['bad','good']]
    report = download({'sources':specs,'raw_dir':str(tmp_path),'local_dir':str(tmp_path/'absent')})
    assert report['bad']['status'] == 'unavailable'
    assert report['good']['rows'] == 1
    assert len(list(read_jsonl(tmp_path/'good.jsonl'))) == 1


def test_diagnostics_do_not_fake_human_scores(tmp_path):
    d = diagnostics('43', '43')
    assert d['exact_reference_match'] is True
    repeated = diagnostics('один два три один два три один два три')
    assert repeated['repeated_trigram_fraction'] > 0
    result = {'category':'math','prompt':'19+24','response':'43','diagnostics':d,'human_score':None}
    report = save_report([result], tmp_path/'report', 'test')
    assert report['results'][0]['human_score'] is None
    assert (tmp_path/'report.json').is_file() and (tmp_path/'report.md').is_file()
