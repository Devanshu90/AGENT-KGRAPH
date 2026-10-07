from agentkgraph.evaluation.metrics import (
    exact_match,
    token_f1,
    hits_at_1,
    precision_at_k,
    provenance_completeness,
    routing_accuracy,
)


def test_exact_match():
    assert exact_match("Arthur's Magazine",["Arthur's Magazine"])==1.0
    assert exact_match("wrong",["Arthur's Magazine"])==0.0


def test_token_f1():
    assert token_f1("Arthur's Magazine",["Arthur's Magazine"])==1.0
    assert token_f1("Arthur Magazine",["Arthur's Magazine"])>0.0


def test_hits_at_1():
    assert hits_at_1(
        ["Arthur's Magazine","First for Women"],
        ["Arthur's Magazine"]
    )==1.0

    assert hits_at_1(
        ["First for Women","Arthur's Magazine"],
        ["Arthur's Magazine"]
    )==0.0


def test_precision_at_k():
    assert precision_at_k(
        ["a","b","c"],
        ["a","c"],
        3
    )==2/3

    assert precision_at_k(
        ["a","b","c"],
        ["a"],
        1
    )==1.0


def test_provenance_completeness():
    assert provenance_completeness(["p1","p2"],2)==1.0
    assert provenance_completeness(["p1"],2)==0.5
    assert provenance_completeness([],2)==0.0


def test_routing_accuracy():
    assert routing_accuracy(
        ["kg","vector","hybrid"],
        ["kg","vector","hybrid"]
    )==1.0

    assert routing_accuracy(
        ["kg","vector","kg"],
        ["kg","hybrid","kg"]
    )==2/3