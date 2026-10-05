import re


def extract_entity(question):
    match=re.search(r"\[([^\]]+)\]",question)

    if match:
        return match.group(1).strip()

    return None


def infer_chain(question):
    q=question.lower()

    if "same director" in q:
        return [
            ("in","directed_by"),
            ("out","directed_by")
        ]

    if "same screenwriter" in q:
        return [
            ("in","written_by"),
            ("out","written_by")
        ]

    if "person wrote the films directed" in q:
        return [
            ("in","directed_by"),
            ("out","written_by")
        ]

    if "actors in the movies written" in q:
        return [
            ("in","written_by"),
            ("out","starred_actors")
        ]

    if "director of" in q and "acted films" in q:
        return [
            ("in","starred_actors"),
            ("out","directed_by")
        ]

    if "co-wrote movies with" in q:
        return [
            ("in","written_by"),
            ("out","written_by")
        ]

    if "genres do the movies written" in q:
        return [
            ("in","written_by"),
            ("out","has_genre")
        ]

    if "genres do the movies directed" in q:
        return [
            ("in","directed_by"),
            ("out","has_genre")
        ]

    if "release years of the movies acted" in q:
        return [
            ("in","starred_actors"),
            ("out","release_year")
        ]

    if "languages are the films starred" in q:
        return [
            ("in","starred_actors"),
            ("out","has_language")
        ]

    if "movies written by" in q:
        return [
            ("in","written_by")
        ]

    if "movies directed by" in q:
        return [
            ("in","directed_by")
        ]

    if "movies acted by" in q:
        return [
            ("in","starred_actors")
        ]

    if "films directed by" in q:
        return [
            ("in","directed_by")
        ]

    if "films written by" in q:
        return [
            ("in","written_by")
        ]

    return []


def parse_query(question,hop):
    entity=extract_entity(question)
    chain=infer_chain(question)

    return {
        "entity":entity,
        "hop":hop,
        "relations":chain[:hop]
    }