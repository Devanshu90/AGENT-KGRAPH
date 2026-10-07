from agentkgraph.data.dataset_integration import DatasetIntegration
from agentkgraph.evaluation.dataset_evaluator import DatasetEvaluator


def test_evaluate_records():
    loader=DatasetIntegration()
    evaluator=DatasetEvaluator(loader)

    records=loader.load_metaqa("train")[:2]

    predictions=[
        records[0].answers[0],
        "incorrect answer",
    ]

    result=evaluator.evaluate_records(
        records,
        predictions,
    )

    assert result["count"]==2
    assert 0.0<=result["exact_match"]<=1.0
    assert 0.0<=result["token_f1"]<=1.0


def test_metaqa_hop_metrics():
    loader=DatasetIntegration()
    evaluator=DatasetEvaluator(loader)

    records=loader.load_metaqa("train")[:20]

    predictions=[
        record.answers[0]
        for record in records
    ]

    result=evaluator.evaluate_dataset(
        "MetaQA",
        "train",
        predictions,
        limit=20,
    )

    assert result["dataset"]=="MetaQA"
    assert result["split"]=="train"
    assert result["count"]==20
    assert result["exact_match"]==1.0
    assert result["token_f1"]==1.0
    assert result["hop_metrics"]


def test_hotpotqa_evaluation():
    loader=DatasetIntegration()
    evaluator=DatasetEvaluator(loader)

    records=loader.load_hotpotqa("train")[:10]

    predictions=[
        record.answers[0]
        for record in records
    ]

    result=evaluator.evaluate_dataset(
        "HotpotQA",
        "train",
        predictions,
        limit=10,
    )

    assert result["dataset"]=="HotpotQA"
    assert result["split"]=="train"
    assert result["count"]==10
    assert result["exact_match"]==1.0
    assert result["token_f1"]==1.0


def test_webqsp_evaluation():
    loader=DatasetIntegration()
    evaluator=DatasetEvaluator(loader)

    records=loader.load_webqsp("train")[:10]

    predictions=[
        record.answers[0]
        for record in records
    ]

    result=evaluator.evaluate_dataset(
        "WebQSP",
        "train",
        predictions,
        limit=10,
    )

    assert result["dataset"]=="WebQSP"
    assert result["split"]=="train"
    assert result["count"]==10
    assert result["exact_match"]==1.0
    assert result["token_f1"]==1.0


def test_fixed_prediction():
    evaluator=DatasetEvaluator()

    result=evaluator.evaluate_fixed_prediction(
        "MetaQA",
        "train",
        "not the correct answer",
        limit=5,
    )

    assert result["count"]==5
    assert result["exact_match"]==0.0


def test_prediction_length_validation():
    evaluator=DatasetEvaluator()

    records=evaluator.loader.load_metaqa("train")[:2]

    try:
        evaluator.evaluate_records(
            records,
            ["only one prediction"],
        )
        assert False
    except ValueError:
        assert True