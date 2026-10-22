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
        add_triple(kg, sub, rel, obj)

    return kg


def add_triple(kg, sub, rel, obj):
    kg.add_entity(sub)
    kg.add_entity(obj)
    kg.graph.add_edge(
        sub,
        obj,
        predicate=rel,
        confidence=1.0,
    )


def test_filmography_direct_path_is_retrieved():
    retriever = KGRetriever(make_kg())

    results = retriever.search(
        "what films did [Michelle Trachtenberg] star in"
    )

    direct_paths = [
        item for item in results
        if len(item["path"]) == 2
        and item["path"][0] == (None, "Michelle Trachtenberg")
        and item["path"][1] == ("starred_actors", "Ice Princess")
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


def test_filmography_self_loop_is_retrieved():
    kg = make_kg()
    add_triple(kg, "Lemmy", "starred_actors", "Lemmy")

    retriever = KGRetriever(kg)
    results = retriever.search(
        "what films did [Lemmy] act in"
    )

    assert any(
        len(item["path"]) == 2
        and item["path"][0] == (None, "Lemmy")
        and item["path"][1] == ("starred_actors", "Lemmy")
        for item in results
    )


def test_mirror_has_two_faces_director_chain_is_retrieved():
    kg = make_kg()

    add_triple(
        kg,
        "The Mirror Has Two Faces",
        "directed_by",
        "Barbra Streisand",
    )
    add_triple(
        kg,
        "The Mirror Has Two Faces",
        "starred_actors",
        "Barbra Streisand",
    )
    add_triple(
        kg,
        "The Prince of Tides",
        "directed_by",
        "Barbra Streisand",
    )
    add_triple(
        kg,
        "The Prince of Tides",
        "starred_actors",
        "Nick Nolte",
    )

    retriever = KGRetriever(kg)
    query = (
        "who are the actors in the films directed by "
        "the director of [The Mirror Has Two Faces]"
    )
    results = retriever.search(query)
    paths = [item["path"] for item in results]

    assert [
        (None, "The Mirror Has Two Faces"),
        ("starred_actors", "Barbra Streisand"),
    ] in paths

    assert [
        (None, "The Mirror Has Two Faces"),
        ("directed_by", "Barbra Streisand"),
        ("directed_by (reverse)", "The Prince of Tides"),
        ("starred_actors", "Nick Nolte"),
    ] in paths