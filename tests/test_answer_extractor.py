
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


def test_filmography_does_not_include_co_star_movies():
    evidence = [
        {
            "path_id": "direct",
            "path": [
                (None, "Joe Thomas"),
                ("starred_actors (reverse)", "The Inbetweeners 2"),
                ("starred_actors", "James Buckley"),
                ("starred_actors (reverse)", "The Pyramid"),
            ],
        },
    ]

    result = KGPathAnswerExtractor().extract(
        "Joe Thomas appears in which movies",
        evidence,
    )

    assert result is not None
    assert result["answer"] == "The Inbetweeners 2"


def test_language_target_overrides_shared_actor_intermediate_movie():
    evidence = [
        {
            "path_id": "p1",
            "path": [
                (None, "Creepshow"),
                ("starred_actors", "Leslie Nielsen"),
                ("starred_actors (reverse)", "Mr. Magoo"),
                ("in_language", "English"),
            ],
        },
        {
            "path_id": "p2",
            "path": [
                (None, "Creepshow"),
                ("starred_actors", "Hal Holbrook"),
                ("starred_actors (reverse)", "The Star Chamber"),
                ("in_language", "Polish"),
            ],
        },
    ]

    result = KGPathAnswerExtractor().extract(
        "the films that share actors with the film [Creepshow] were in which languages",
        evidence,
    )

    assert result is not None
    assert result["answer"] == "English, Polish"


def test_genre_target_filters_to_shared_director_paths():
    evidence = [
        {
            "path_id": "p1",
            "path": [
                (None, "The Dance of Reality"),
                ("directed_by", "Alejandro Jodorowsky"),
                ("directed_by (reverse)", "The Holy Mountain"),
                ("has_genre", "Drama"),
            ],
        },
        {
            "path_id": "p2",
            "path": [
                (None, "The Dance of Reality"),
                ("directed_by", "Alejandro Jodorowsky"),
                ("directed_by (reverse)", "El Topo"),
                ("has_genre", "Western"),
            ],
        },
        {
            "path_id": "bad",
            "path": [
                (None, "The Dance of Reality"),
                ("written_by (reverse)", "Unrelated Film"),
                ("has_genre", "Comedy"),
            ],
        },
    ]

    result = KGPathAnswerExtractor().extract(
        "the films that share directors with [The Dance of Reality] fall under which genres",
        evidence,
    )

    assert result is not None
    assert result["answer"] == "Drama, Western"


def test_directors_of_movies_written_by_writer_of_she():
    evidence = [
        {
            "path_id": "correct",
            "path": [
                (None, "She"),
                ("written_by", "H. Rider Haggard"),
                ("written_by (reverse)", "Allan Quatermain and the Lost City of Gold"),
                ("directed_by", "Gary Nelson"),
            ],
        },
        {
            "path_id": "wrong",
            "path": [
                (None, "She"),
                ("directed_by", "Robert Day"),
                ("directed_by (reverse)", "The Quick and the Dead"),
                ("directed_by", "Sam Raimi"),
            ],
        },
    ]

    result = KGPathAnswerExtractor().extract(
        "who are the directors of the movies written by the writer of [She]",
        evidence,
    )

    assert result is not None
    assert result["answer"] == "Gary Nelson"
    assert result["citations"] == ["correct"]


def test_shared_writer_release_years_ignore_unrelated_paths():
    evidence = [
        {
            "path_id": "p1",
            "path": [
                (None, "Grown Ups 2"),
                ("written_by", "Adam Sandler"),
                ("written_by (reverse)", "Billy Madison"),
                ("release_year", "1995"),
            ],
        },
        {
            "path_id": "p2",
            "path": [
                (None, "Grown Ups 2"),
                ("written_by", "Adam Sandler"),
                ("written_by (reverse)", "Eight Crazy Nights"),
                ("release_year", "2002"),
            ],
        },
        {
            "path_id": "bad",
            "path": [
                (None, "Grown Ups 2"),
                ("directed_by", "Unrelated Director"),
                ("directed_by (reverse)", "Unrelated Film"),
                ("release_year", "1962"),
            ],
        },
    ]

    result = KGPathAnswerExtractor().extract(
        "what were the release years of the films that share writers with the film [Grown Ups 2]",
        evidence,
    )

    assert result is not None
    assert result["answer"] == "1995, 2002"


def test_shared_movie_query_returns_movie_not_actor():
    evidence = [
        {
            "path_id": "p1",
            "path": [
                (None, "Jack the Bear"),
                ("starred_actors", "Danny DeVito"),
                ("starred_actors (reverse)", "The Rainmaker"),
            ],
        }
    ]

    result = KGPathAnswerExtractor().extract(
        "which movies share actors with [Jack the Bear]",
        evidence,
    )

    assert result is not None
    assert result["answer"] == "The Rainmaker"


def test_extract_language_before_irrelevant_reverse_edge():
    evidence = [
        {
            "path_id": "language_path",
            "path": [
                (None, "David Mandel"),
                ("directed_by (reverse)", "EuroTrip"),
                ("in_language", "German"),
                ("in_language (reverse)", "Operation Crossbow"),
            ],
        }
    ]

    result = KGPathAnswerExtractor().extract(
        "what are the primary languages in the movies directed by [David Mandel]",
        evidence,
    )

    assert result is not None
    assert result["answer"] == "German"
    assert result["citations"] == ["language_path"]
