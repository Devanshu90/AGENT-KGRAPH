from agentkgraph.evaluation.ablation import AblationEvaluator


def test_ablation_evaluation():
    evaluator=AblationEvaluator()

    configurations={
        "full":{
            "exact_match":0.80,
            "token_f1":0.85,
            "hits_at_1":0.90,
        },
        "vector_only":{
            "exact_match":0.60,
            "token_f1":0.70,
            "hits_at_1":0.65,
        },
        "kg_only":{
            "exact_match":0.70,
            "token_f1":0.75,
            "hits_at_1":0.80,
        },
        "without_evolution":{
            "exact_match":0.75,
            "token_f1":0.80,
            "hits_at_1":0.85,
        },
    }

    result=evaluator.evaluate(configurations)

    assert result["baseline"]=="full"
    assert result["deltas"]["vector_only"]["exact_match"]==-0.2
    assert result["deltas"]["kg_only"]["hits_at_1"]==-0.1
    assert result["deltas"]["without_evolution"]["token_f1"]==-0.05


def test_ranking():
    evaluator=AblationEvaluator()

    configurations={
        "full":{"exact_match":0.80},
        "vector_only":{"exact_match":0.60},
        "kg_only":{"exact_match":0.70},
    }

    ranking=evaluator.ranking(
        configurations,
        "exact_match",
    )

    assert ranking[0]==("full",0.80)
    assert ranking[1]==("kg_only",0.70)
    assert ranking[2]==("vector_only",0.60)


def test_best_configuration():
    evaluator=AblationEvaluator()

    configurations={
        "full":{"token_f1":0.85},
        "vector_only":{"token_f1":0.70},
        "kg_only":{"token_f1":0.80},
    }

    assert evaluator.best_configuration(
        configurations,
        "token_f1",
    )=="full"


def test_metric_impact():
    evaluator=AblationEvaluator()

    configurations={
        "full":{"hits_at_1":0.90},
        "vector_only":{"hits_at_1":0.65},
    }

    impact=evaluator.metric_impact(
        configurations,
        "vector_only",
        "hits_at_1",
    )

    assert impact==-0.25


def test_missing_baseline():
    evaluator=AblationEvaluator(
        baseline="missing"
    )

    try:
        evaluator.evaluate({
            "full":{"exact_match":0.8}
        })
        assert False
    except ValueError:
        assert True