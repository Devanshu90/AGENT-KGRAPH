from agentkgraph.evaluation.runner import EvaluationRunner


def test_evaluate_example():
    runner=EvaluationRunner(k=3)

    result=runner.evaluate_example(
        prediction="Arthur's Magazine",
        references=["Arthur's Magazine"],
        retrieved=["Arthur's Magazine","First for Women"],
        provenance=["p1","p2"],
        required_provenance=2,
        predicted_route="hybrid",
        expected_route="hybrid",
    )

    assert result["exact_match"]==1.0
    assert result["token_f1"]==1.0
    assert result["hits_at_1"]==1.0
    assert result["precision_at_k"]==0.5
    assert result["provenance_completeness"]==1.0
    assert result["routing_accuracy"]==1.0


def test_evaluate_dataset():
    runner=EvaluationRunner(k=2)

    examples=[
        {
            "prediction":"Arthur's Magazine",
            "references":["Arthur's Magazine"],
            "retrieved":["Arthur's Magazine","First for Women"],
            "provenance":["p1"],
            "required_provenance":1,
            "predicted_route":"hybrid",
            "expected_route":"hybrid",
        },
        {
            "prediction":"wrong",
            "references":["First for Women"],
            "retrieved":["wrong","First for Women"],
            "provenance":[],
            "required_provenance":1,
            "predicted_route":"vector",
            "expected_route":"kg",
        },
    ]

    result=runner.evaluate(examples)

    assert result["count"]==2
    assert result["exact_match"]==0.5
    assert result["hits_at_1"]==0.5
    assert result["routing_accuracy"]==0.5
    assert result["provenance_completeness"]==0.5


def test_empty_evaluation():
    runner=EvaluationRunner()

    result=runner.evaluate([])

    assert result["count"]==0
    assert result["exact_match"]==0.0
    assert result["token_f1"]==0.0


def test_invalid_k():
    try:
        EvaluationRunner(k=0)
        assert False
    except ValueError:
        assert True