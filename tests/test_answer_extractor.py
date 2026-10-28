
from agentkgraph.retrieval.answer_extractor import KGPathAnswerExtractor


def test_same_movie_question_extracts_shared_actors():
    ex = KGPathAnswerExtractor()
    evidence = [
        {
            "path_id": "p1",
            "path": [
                [None, "Angie Everhart"],
                ["starred_actors (reverse)", "Bordello of Blood"],
                ["starred_actors", "Dennis Miller"],
                ["starred_actors (reverse)", "Murder at 1600"],
            ],
        },
        {
            "path_id": "p2",
            "path": [
                [None, "Angie Everhart"],
                ["starred_actors (reverse)", "Bordello of Blood"],
                ["starred_actors", "Erika Eleniak"],
                ["starred_actors (reverse)", "Girl in the Cadillac"],
            ],
        },
    ]

    result = ex.extract(
        "who appeared in the same movie with [Angie Everhart]",
        evidence,
    )

    assert result is not None
    assert "Dennis Miller" in result["answer"]
    assert "Erika Eleniak" in result["answer"]


def test_actor_movie_question_extracts_directors():
    ex = KGPathAnswerExtractor()
    evidence = [
        {
            "path_id": "p1",
            "path": [
                [None, "Dorothy Malone"],
                ["starred_actors (reverse)", "Colorado Territory"],
                ["directed_by", "Raoul Walsh"],
                ["directed_by (reverse)", "Distant Drums"],
            ],
        },
    ]

    result = ex.extract(
        "the movies starred by [Dorothy Malone] were directed by who",
        evidence,
    )

    assert result is not None
    assert "Raoul Walsh" in result["answer"]


def test_director_filmography_extracts_other_movies():
    ex = KGPathAnswerExtractor()
    evidence = [
        {
            "path_id": "p1",
            "path": [
                [None, "JFK"],
                ["directed_by", "Oliver Stone"],
                ["directed_by (reverse)", "World Trade Center"],
            ],
        },
        {
            "path_id": "p2",
            "path": [
                [None, "JFK"],
                ["directed_by", "Oliver Stone"],
                ["directed_by (reverse)", "Nixon"],
            ],
        },
    ]

    result = ex.extract(
        "the director of [JFK] also directed which movies",
        evidence,
    )

    assert result is not None
    assert "World Trade Center" in result["answer"]
    assert "Nixon" in result["answer"]


def test_genre_extraction_does_not_use_unrelated_reverse_edge():
    ex = KGPathAnswerExtractor()
    evidence = [
        {
            "path_id": "p1",
            "path": [
                [None, "Mike Nichols"],
                ["directed_by (reverse)", "Regarding Henry"],
                ["has_genre", "Drama"],
                ["has_genre (reverse)", "Dirty Filthy Love"],
            ],
        },
    ]

    result = ex.extract(
        "what are the genres of the films directed by [Mike Nichols]",
        evidence,
    )

    assert result is not None
    assert result["answer"] == "Drama"
    assert "Dirty Filthy Love" not in result["answer"]


def test_shared_director_question_extracts_languages_not_movies():
    ex = KGPathAnswerExtractor()
    evidence = [
        {
            "path_id": "p1",
            "path": [
                [None, "Catch Me If You Can"],
                ["directed_by", "Steven Spielberg"],
                ["directed_by (reverse)", "Amistad"],
                ["in_language", "Mende"],
            ],
        },
    ]

    result = ex.extract(
        "the films that share directors with the film [Catch Me If You Can] were in which languages",
        evidence,
    )

    assert result is not None
    assert result["answer"] == "Mende"
    assert "Amistad" not in result["answer"]


def test_shared_director_question_extracts_genres_not_movies():
    ex = KGPathAnswerExtractor()
    evidence = [
        {
            "path_id": "p1",
            "path": [
                [None, "The Dance of Reality"],
                ["directed_by", "Alejandro Jodorowsky"],
                ["directed_by (reverse)", "The Holy Mountain"],
                ["has_genre", "Drama"],
            ],
        },
        {
            "path_id": "p2",
            "path": [
                [None, "The Dance of Reality"],
                ["directed_by", "Alejandro Jodorowsky"],
                ["directed_by (reverse)", "El Topo"],
                ["has_genre", "Western"],
            ],
        },
    ]

    result = ex.extract(
        "the films that share directors with the film [The Dance of Reality] were in which genres",
        evidence,
    )

    assert result is not None
    assert result["answer"] == "Drama, Western"
    assert "The Holy Mountain" not in result["answer"]
    assert "El Topo" not in result["answer"]


def test_shared_writer_question_extracts_release_years_not_movies():
    ex = KGPathAnswerExtractor()
    evidence = [
        {
            "path_id": "p1",
            "path": [
                [None, "Grown Ups 2"],
                ["written_by", "Adam Sandler"],
                ["written_by (reverse)", "Billy Madison"],
                ["release_year", "1995"],
            ],
        },
    ]

    result = ex.extract(
        "what were the release years of the films that share writers with [Grown Ups 2]",
        evidence,
    )

    assert result is not None
    assert result["answer"] == "1995"
    assert "Billy Madison" not in result["answer"]


def test_shared_actor_question_extracts_languages():
    ex = KGPathAnswerExtractor()
    evidence = [
        {
            "path_id": "p1",
            "path": [
                [None, "Waxworks"],
                ["starred_actors", "Conrad Veidt"],
                ["starred_actors (reverse)", "The Man Who Laughs"],
                ["in_language", "French"],
            ],
        },
    ]

    result = ex.extract(
        "the movies that share actors with the movie [Waxworks] were in which languages",
        evidence,
    )

    assert result is not None
    assert result["answer"] == "French"


def test_shared_actor_question_extracts_directors_of_other_movies():
    ex = KGPathAnswerExtractor()
    evidence = [
        {
            "path_id": "p1",
            "path": [
                [None, "Buchanan Rides Alone"],
                ["starred_actors", "Randolph Scott"],
                ["starred_actors (reverse)", "The Last of the Mohicans"],
                ["directed_by", "George B. Seitz"],
            ],
        },
    ]

    result = ex.extract(
        "the films that share actors with the film [Buchanan Rides Alone] are directed by who",
        evidence,
    )

    assert result is not None
    assert result["answer"] == "George B. Seitz"
    assert "The Last of the Mohicans" not in result["answer"]


def test_writer_path_extracts_directors_of_movies_written_by_writer():
    ex = KGPathAnswerExtractor()
    evidence = [
        {
            "path_id": "p1",
            "path": [
                [None, "She"],
                ["written_by", "H. Rider Haggard"],
                ["written_by (reverse)", "Allan Quatermain and the Lost City of Gold"],
                ["directed_by", "Gary Nelson"],
            ],
        },
        {
            "path_id": "p2",
            "path": [
                [None, "She"],
                ["directed_by", "Robert Day"],
                ["directed_by (reverse)", "Unrelated Movie"],
                ["directed_by", "Someone Else"],
            ],
        },
    ]

    result = ex.extract(
        "the director of the films written by [H. Rider Haggard] was who",
        evidence,
    )

    assert result is not None
    assert "Gary Nelson" in result["answer"]
    assert "Someone Else" not in result["answer"]


def test_language_question_with_spoken_in_phrasing():
    ex = KGPathAnswerExtractor()
    assert ex._target_relation(
        "what are the languages spoken in the movies whose actors also appear in [Hard Rain]"
    ) == "in_language"


def test_genre_question_with_genres_of_phrasing():
    ex = KGPathAnswerExtractor()
    assert ex._target_relation(
        "what are the genres of the movies whose actors also appear in [Scarface]"
    ) == "has_genre"
