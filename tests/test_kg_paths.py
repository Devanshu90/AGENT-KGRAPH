from agentkgraph.config import Config
from agentkgraph.kg.graph_store import GraphStore
from agentkgraph.retrieval.kg_paths import KGRetriever


def make_kg():
    cfg = Config()
    cfg.kg.hop_limit = 3
    kg = GraphStore(cfg.kg)

    triples = [
        ("Michelle Trachtenberg", "starred_actors", "Ice Princess"),
        ("Michelle Trachtenberg", "has_tags", "17 Again"),
        ("17 Again", "starred_actors", "Leslie Mann"),
        ("Leslie Mann", "starred_actors", "Knocked Up"),
    ]

    for sub, rel, obj in triples:
        kg.add_entity(sub)
        kg.add_entity(obj)
        kg.graph.add_edge(
            sub,
            obj,
            predicate=rel,
            confidence=1.0,
        )

    return kg


def test_filmography_direct_path_is_retrieved():
    retriever = KGRetriever(make_kg())

    results = retriever.search(
        "what films did [Michelle Trachtenberg] star in"
    )

    direct_paths = [
        item for item in results
        if len(item["path"]) == 2
        and item["path"][0] == (None, "Michelle Trachtenberg")
        and item["path"][1] == (
            "starred_actors",
            "Ice Princess",
        )
    ]

    assert direct_paths


def test_direct_filmography_path_ranks_above_unrelated_tag_path():
    retriever = KGRetriever(make_kg())
    query = "what films did [Michelle Trachtenberg] star in"

    relevant = retriever._relevant_relations(query)
    target = retriever._target_relations(query)

    direct = {
        "path": [
            (None, "Michelle Trachtenberg"),
            ("starred_actors", "Ice Princess"),
        ],
        "score": 1.0,
    }

    unrelated = {
        "path": [
            (None, "Michelle Trachtenberg"),
            ("has_tags (reverse)", "17 Again"),
        ],
        "score": 1.0,
    }

    assert retriever._rank_path(
        direct, relevant, target
    ) > retriever._rank_path(
        unrelated, relevant, target
    )
