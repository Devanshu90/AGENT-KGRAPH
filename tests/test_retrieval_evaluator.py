from agentkgraph.evaluation.retrieval_evaluator import RetrievalEvaluator


def test_evaluate_example():
    evaluator=RetrievalEvaluator(k=3)

    result=evaluator.evaluate_example(
        retrieved=[
            "Arthur's Magazine",
            "First for Women",
            "Time",
        ],
        references=[
            "Arthur's Magazine",
        ],
        predicted_route="hybrid",
        expected_route="hybrid",
    )

    assert result["hits_at_1"]==1.0
    assert result["precision_at_k"]==1/3
    assert result["routing_accuracy"]==1.0


def test_evaluate():
    evaluator=RetrievalEvaluator(k=2)

    examples=[
        {
            "retrieved":["a","b"],
            "references":["a"],
            "predicted_route":"kg",
            "expected_route":"kg",
        },
        {
            "retrieved":["x","a"],
            "references":["a"],
            "predicted_route":"vector",
            "expected_route":"kg",
        },
    ]

    result=evaluator.evaluate(examples)

    assert result["count"]==2
    assert result["hits_at_1"]==0.5
    assert result["precision_at_k"]==0.5
    assert result["routing_accuracy"]==0.5


def test_compare_methods():
    evaluator=RetrievalEvaluator(k=2)

    examples={
        "vector":[
            {
                "retrieved":["a","x"],
                "references":["a"],
            }
        ],
        "kg":[
            {
                "retrieved":["x","a"],
                "references":["a"],
            }
        ],
        "hybrid":[
            {
                "retrieved":["a","x"],
                "references":["a"],
            }
        ],
    }

    result=evaluator.compare_methods(examples)

    assert set(result)=={"vector","kg","hybrid"}
    assert result["vector"]["hits_at_1"]==1.0
    assert result["kg"]["hits_at_1"]==0.0
    assert result["hybrid"]["hits_at_1"]==1.0


def test_empty_evaluation():
    evaluator=RetrievalEvaluator()

    result=evaluator.evaluate([])

    assert result["count"]==0
    assert result["hits_at_1"]==0.0
    assert result["precision_at_k"]==0.0


def test_invalid_k():
    try:
        RetrievalEvaluator(k=0)
        assert False
    except ValueError:
        assert True