from agentkgraph.config import Config
from agentkgraph.pipeline import Engine
from agentkgraph.agents.extractor import Extractor
from agentkgraph.agents.verifier import Verifier
from agentkgraph.retrieval.vector import VectorRetriever
from agentkgraph.routing.router import Router
from agentkgraph.generation.synthesizer import Synthesizer


def test_b8_model_configuration():
    config=Config()

    assert config.models.extractor=="Qwen/Qwen2.5-1.5B-Instruct"
    assert config.models.verifier=="cross-encoder/nli-deberta-v3-base"
    assert config.models.router=="Qwen/Qwen2.5-0.5B-Instruct"
    assert config.models.synthesizer=="Qwen/Qwen2.5-3B-Instruct"
    assert config.models.embedder=="BAAI/bge-small-en-v1.5"


def test_b8_extractor_dry_run():
    extractor=Extractor(dry_run=True)

    triples=extractor.extract(
        "Inception was directed by Christopher Nolan.",
        "p1"
    )

    assert len(triples)==1
    assert triples[0].subject=="Inception"
    assert triples[0].predicate=="directed_by"
    assert triples[0].object=="Christopher Nolan"
    assert triples[0].passage_id=="p1"
    assert 0.0<=triples[0].confidence<=1.0


def test_b8_verifier_dry_run():
    extractor=Extractor(dry_run=True)
    verifier=Verifier(dry_run=True)

    triple=extractor.extract(
        "Inception was directed by Christopher Nolan.",
        "p1"
    )[0]

    result=verifier.verify(
        triple,
        "Inception was directed by Christopher Nolan."
    )

    assert result.accepted is True
    assert result.confidence>=0.60


def test_b8_vector_retriever_dry_run():
    passages=[
        {
            "passage_id":"p1",
            "document_id":"d1",
            "text":"Inception was directed by Christopher Nolan."
        },
        {
            "passage_id":"p2",
            "document_id":"d2",
            "text":"Titanic was directed by James Cameron."
        }
    ]

    retriever=VectorRetriever(
        passages,
        top_k=2,
        dry_run=True
    )

    results=retriever.search(
        "Christopher Nolan directed Inception"
    )

    assert len(results)==2
    assert results[0]["passage_id"]=="p1"


def test_b8_router_dry_run():
    router=Router(dry_run=True)

    assert router.route(
        "Who directed Inception?"
    )=="hybrid"

    assert router.route(
        "When was Inception released?"
    )=="hybrid"


def test_b8_synthesizer_dry_run():
    synthesizer=Synthesizer(dry_run=True)

    evidence=[
        {
            "passage_id":"p1",
            "text":"Inception was directed by Christopher Nolan."
        }
    ]

    result=synthesizer.generate(
        "Who directed Inception?",
        evidence
    )

    assert result["answer"]=="Inception was directed by Christopher Nolan."
    assert result["sources"]==["p1"]


def test_b8_engine_integration():
    passages=[
        {
            "passage_id":"p1",
            "document_id":"d1",
            "text":"Inception was directed by Christopher Nolan."
        }
    ]

    engine=Engine(
        passages=passages,
        dry_run=True
    )

    engine._load_evolution_components()

    assert engine.extractor is not None
    assert engine.verifier is not None
    assert engine.merger is not None
    assert engine.synthesizer is not None
    assert engine.feedback is not None

    triple=engine.extractor.extract(
        "Inception was directed by Christopher Nolan.",
        "p1"
    )[0]

    result=engine.verifier.verify(
        triple,
        "Inception was directed by Christopher Nolan."
    )

    assert result.accepted is True

    engine.unload_all()
