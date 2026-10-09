from agentkgraph.retrieval.answer_extractor import KGPathAnswerExtractor


def test_extract_directors_from_starred_movies():
    evidence = [
        {
            "path_id": "kg_path_0001",
            "path": [
                (None, "John Krasinski"),
                ("starred_actors (reverse)", "A Movie"),
                ("directed_by", "Director One"),
            ],
        },
        {
            "path_id": "kg_path_0002",
            "path": [
                (None, "John Krasinski"),
                ("starred_actors (reverse)", "Another Movie"),
                ("directed_by", "Director Two"),
            ],
        },
    ]

    result = KGPathAnswerExtractor().extract(
        "which person directed the movies starred by [John Krasinski]",
        evidence,
    )

    assert result is not None
    assert result["answer"] == "Director One, Director Two"
    assert result["citations"] == ["kg_path_0001", "kg_path_0002"]


def test_extract_languages_from_shared_director_paths():
    evidence = [
        {
            "path_id": "kg_path_0001",
            "path": [
                (None, "Catch Me If You Can"),
                ("directed_by", "Director One"),
                ("directed_by (reverse)", "Film One"),
                ("in_language", "German"),
            ],
        },
        {
            "path_id": "kg_path_0002",
            "path": [
                (None, "Catch Me If You Can"),
                ("directed_by", "Director One"),
                ("directed_by (reverse)", "Film Two"),
                ("in_language", "Polish"),
            ],
        },
    ]

    result = KGPathAnswerExtractor().extract(
        "the films that share directors with the film [Catch Me If You Can] were in which languages",
        evidence,
    )

    assert result is not None
    assert result["answer"] == "German, Polish"


def test_extract_actors_from_director_movie_paths():
    evidence = [
        {
            "path_id": "kg_path_0001",
            "path": [
                (None, "Written on the Wind"),
                ("directed_by", "Douglas Sirk"),
                ("directed_by (reverse)", "Sleep, My Love"),
                ("starred_actors", "Robert Cummings"),
            ],
        },
        {
            "path_id": "kg_path_0002",
            "path": [
                (None, "Written on the Wind"),
                ("directed_by", "Douglas Sirk"),
                ("directed_by (reverse)", "Sleep, My Love"),
                ("starred_actors", "Don Ameche"),
            ],
        },
    ]

    result = KGPathAnswerExtractor().extract(
        "who starred movies for the director of [Written on the Wind]",
        evidence,
    )

    assert result is not None
    assert result["answer"] == "Robert Cummings, Don Ameche"


def test_no_matching_target_relation_returns_none():
    evidence = [
        {
            "path_id": "kg_path_0001",
            "path": [
                (None, "Some Movie"),
                ("directed_by", "Some Director"),
            ],
        },
    ]

    result = KGPathAnswerExtractor().extract(
        "which languages were the movies in",
        evidence,
    )

    assert result is None

def test_extract_movies_for_person_from_reverse_starred_actors_path():
    extractor = KGPathAnswerExtractor()
    query = "what does Gregoire Colin appear in"
    evidence = [
        {
            "path": [
                (None, "Grégoire Colin"),
                ("starred_actors (reverse)", "Before the Rain"),
            ],
            "path_id": "path_1",
        }
    ]

    result = extractor.extract(query, evidence)

    assert result is not None
    assert result["answer"] == "Before the Rain"
    assert result["citations"] == ["path_1"]


def test_extract_multiple_movies_for_person_from_reverse_paths():
    extractor = KGPathAnswerExtractor()
    query = "Joe Thomas appears in which movies"
    evidence = [
        {
            "path": [
                (None, "Joe Thomas"),
                ("starred_actors (reverse)", "The Inbetweeners Movie"),
            ],
            "path_id": "path_1",
        },
        {
            "path": [
                (None, "Joe Thomas"),
                ("starred_actors (reverse)", "The Inbetweeners 2"),
            ],
            "path_id": "path_2",
        },
    ]

    result = extractor.extract(query, evidence)

    assert result is not None
    assert result["answer"] == "The Inbetweeners Movie, The Inbetweeners 2"
    assert result["citations"] == ["path_1", "path_2"]



def test_extract_co_directors_from_reverse_director_paths():
    evidence = [
        {
            "path_id": "kg_path_0001",
            "path": [
                (None, "Delbert Mann"),
                ("directed_by (reverse)", "Film One"),
                ("directed_by", "Franco Zeffirelli"),
            ],
        },
        {
            "path_id": "kg_path_0002",
            "path": [
                (None, "Delbert Mann"),
                ("directed_by (reverse)", "Film Two"),
                ("directed_by", "Cary Fukunaga"),
            ],
        },
    ]

    result = KGPathAnswerExtractor().extract(
        "who are movie co-directors of [Delbert Mann]",
        evidence,
    )

    assert result is not None
    assert result["answer"] == "Franco Zeffirelli, Cary Fukunaga"
    assert result["citations"] == ["kg_path_0001", "kg_path_0002"]
