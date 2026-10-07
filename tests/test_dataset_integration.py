from agentkgraph.data.dataset_integration import DatasetIntegration


def test_metaqa_integration():
    loader=DatasetIntegration()

    records=loader.load_metaqa("train")

    assert len(records)>0
    assert records[0].dataset=="MetaQA"
    assert records[0].split=="train"
    assert records[0].question
    assert isinstance(records[0].answers,list)
    assert records[0].hop in {1,2,3}


def test_hotpotqa_integration():
    loader=DatasetIntegration()

    records=loader.load_hotpotqa("train")

    assert len(records)>0
    assert records[0].dataset=="HotpotQA"
    assert records[0].split=="train"
    assert records[0].question
    assert isinstance(records[0].answers,list)
    assert isinstance(records[0].supporting_facts,list)


def test_webqsp_integration():
    loader=DatasetIntegration()

    records=loader.load_webqsp("train")

    assert len(records)>0
    assert records[0].dataset=="WebQSP"
    assert records[0].split=="train"
    assert records[0].question
    assert isinstance(records[0].answers,list)
    assert isinstance(records[0].parses,list)


def test_dataset_dispatch():
    loader=DatasetIntegration()

    assert len(loader.load("MetaQA","train"))>0
    assert len(loader.load("HotpotQA","train"))>0
    assert len(loader.load("WebQSP","train"))>0


def test_metaqa_kg_loading():
    loader=DatasetIntegration()

    triples=loader.load_metaqa_kg()

    assert len(triples)>0
    assert "subject" in triples[0]
    assert "predicate" in triples[0]
    assert "object" in triples[0]


def test_corpus_loading():
    loader=DatasetIntegration()

    passages=loader.load_corpus()

    assert len(passages)>0
    assert "passage_id" in passages[0]
    assert "text" in passages[0]


def test_summary():
    loader=DatasetIntegration()

    summary=loader.summary()

    assert "MetaQA" in summary
    assert "HotpotQA" in summary
    assert "WebQSP" in summary

    assert summary["MetaQA"]["train"]>0
    assert summary["HotpotQA"]["train"]>0
    assert summary["WebQSP"]["train"]>0