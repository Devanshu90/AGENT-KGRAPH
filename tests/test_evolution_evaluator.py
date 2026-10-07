from agentkgraph.evaluation.evolution_evaluator import EvolutionEvaluator


def test_evolution_evaluation():
    evaluator=EvolutionEvaluator()

    before={
        "exact_match":0.40,
        "token_f1":0.50,
        "hits_at_1":0.60,
        "precision_at_k":0.70,
        "provenance_completeness":0.80,
    }

    after={
        "exact_match":0.50,
        "token_f1":0.60,
        "hits_at_1":0.65,
        "precision_at_k":0.75,
        "provenance_completeness":0.90,
    }

    result=evaluator.evaluate(
        before,
        after,
        kg_changes=12,
    )

    assert result["before"]==before
    assert result["after"]==after
    assert result["kg_changes"]==12
    assert result["delta"]["exact_match"]==0.10
    assert result["delta"]["token_f1"]==0.10


def test_improvement():
    evaluator=EvolutionEvaluator()

    before={
        "exact_match":0.50,
        "token_f1":0.50,
    }

    after={
        "exact_match":0.70,
        "token_f1":0.60,
    }

    result=evaluator.improvement(
        before,
        after,
    )

    assert result["exact_match"]==0.20
    assert result["token_f1"]==0.10


def test_improved_and_degraded_metrics():
    evaluator=EvolutionEvaluator()

    before={
        "exact_match":0.70,
        "token_f1":0.50,
        "hits_at_1":0.40,
    }

    after={
        "exact_match":0.60,
        "token_f1":0.70,
        "hits_at_1":0.40,
    }

    assert evaluator.improved_metrics(
        before,
        after,
    )==["token_f1"]

    assert evaluator.degraded_metrics(
        before,
        after,
    )==["exact_match"]


def test_summary():
    evaluator=EvolutionEvaluator()

    before={
        "exact_match":0.40,
        "token_f1":0.50,
    }

    after={
        "exact_match":0.50,
        "token_f1":0.40,
    }

    result=evaluator.summary(
        before,
        after,
        kg_changes=5,
    )

    assert result["summary"]["num_improved"]==1
    assert result["summary"]["num_degraded"]==1
    assert result["summary"]["kg_changes"]==5