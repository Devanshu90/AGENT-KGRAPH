
class KGPathAnswerExtractor:
    def __init__(self):
        self.relation_keywords = {
            "directed_by": ["directed", "director", "directors", "co-director", "codirector"],
            "starred_actors": ["starred", "stars", "star", "actor", "actors", "cast", "appear", "act in"],
            "written_by": ["written", "writer", "writers"],
            "in_language": ["language", "languages"],
            "has_genre": ["genre", "genres"],
            "has_tags": ["tag", "tags"],
            "release_year": ["release year", "released", "when"],
        }

    def _target_relation(self, query):
        q = query.casefold()

        if any(x in q for x in [
            "which languages", "what languages", "in which languages",
            "were in which languages", "primary languages",
            "what language", "which language",
        ]):
            return "in_language"

        if any(x in q for x in [
            "which genres", "what genres", "which genre",
            "what genre", "fall under", "types of",
        ]):
            return "has_genre"

        if any(x in q for x in [
            "release year", "release years", "what year",
            "which year", "when were", "when did",
        ]):
            return "release_year"

        if "tag" in q:
            return "has_tags"

        if any(x in q for x in [
            "who starred", "who stars", "who acted",
            "who appears", "which actors", "what actors",
            "who are the actors", "which actor",
        ]):
            return "starred_actors"

        if any(x in q for x in [
            "who directed", "which director", "what director",
            "who is listed as director", "directors of",
            "directors for", "which directors", "who was the director",
            "co-director", "codirector",
        ]):
            return "directed_by"

        if any(x in q for x in [
            "who wrote", "which writer", "what writer",
            "written by", "screenwriter",
        ]):
            return "written_by"

        shared_terms = [
            "share actor", "share actors", "shared actor", "shared actors",
            "common actor", "common actors",
            "share director", "share directors", "shared director",
            "shared directors", "common director", "common directors",
            "share writer", "share writers", "shared writer",
            "shared writers", "common writer", "common writers",
        ]

        if any(x in q for x in shared_terms):
            return "shared_movies"

        if any(x in q for x in [
            "films did", "what films", "which movies",
            "movies did", "movies was", "appear in",
            "what does", "appears in which movies",
            "starred in which movies", "what movies",
        ]):
            return "filmography"

        if "language" in q:
            return "in_language"
        if "genre" in q:
            return "has_genre"
        if "director" in q or "directed" in q:
            return "directed_by"
        if "writer" in q or "written" in q:
            return "written_by"
        if "actor" in q or "actors" in q or "cast" in q:
            return "starred_actors"

        return None

    def _relations(self, path):
        return [
            str(item[0]).casefold()
            for item in path
            if isinstance(item, (list, tuple))
            and len(item) == 2
            and item[0] is not None
        ]

    def _is_relation(self, relation, target, reverse=None):
        relation = str(relation).casefold()

        if target not in relation:
            return False

        is_reverse = "reverse" in relation
        return reverse is None or is_reverse == reverse

    def _edges(self, path):
        edges = []

        for i, item in enumerate(path):
            if i == 0 or not isinstance(item, (list, tuple)) or len(item) != 2:
                continue

            rel, value = item

            if rel is None or value is None:
                continue

            value = str(value).strip()

            if not value or value.casefold() == "none":
                continue

            edges.append((i, str(rel).casefold(), value))

        return edges

    def _matches_prefix(self, edges, relations):
        if len(edges) < len(relations):
            return False

        return all(
            actual == expected
            for (_, actual, _), expected in zip(edges, relations)
        )

    def _edge_candidates(self, query, path, target):
        q = query.casefold()
        edges = self._edges(path)

        if not edges:
            return []

        shared_actors = any(x in q for x in [
            "share actor", "share actors", "shared actor",
            "shared actors", "common actor", "common actors",
        ])

        shared_directors = any(x in q for x in [
            "share director", "share directors", "shared director",
            "shared directors", "common director", "common directors",
        ])

        shared_writers = any(x in q for x in [
            "share writer", "share writers", "shared writer",
            "shared writers", "common writer", "common writers",
        ])

        if target == "filmography":
            idx, rel, value = edges[0]

            if self._is_relation(rel, "starred_actors", reverse=True):
                return [(idx, value)]

            return []

        if target == "shared_movies":
            if shared_actors:
                pattern = [
                    "starred_actors",
                    "starred_actors (reverse)",
                ]
            elif shared_directors:
                pattern = [
                    "directed_by",
                    "directed_by (reverse)",
                ]
            elif shared_writers:
                pattern = [
                    "written_by",
                    "written_by (reverse)",
                ]
            else:
                return []

            if len(edges) != len(pattern):
                return []

            if self._matches_prefix(edges, pattern):
                return [(edges[-1][0], edges[-1][2])]

            return []

        if target in {
            "in_language", "has_genre", "release_year", "has_tags",
        }:
            if shared_actors:
                prefix = [
                    "starred_actors",
                    "starred_actors (reverse)",
                ]
            elif shared_directors:
                prefix = [
                    "directed_by",
                    "directed_by (reverse)",
                ]
            elif shared_writers:
                prefix = [
                    "written_by",
                    "written_by (reverse)",
                ]
            else:
                prefix = []

            if prefix:
                if len(edges) != len(prefix) + 1:
                    return []

                if not self._matches_prefix(edges, prefix):
                    return []

                final = edges[-1]

                if self._is_relation(final[1], target, reverse=False):
                    return [(final[0], final[2])]

                return []

            final = edges[-1]

            if self._is_relation(final[1], target, reverse=False):
                return [(final[0], final[2])]

            return []

        if target == "directed_by":
            if any(x in q for x in [
                "movies written by the writer",
                "movies written by writer",
                "directors of the movies written",
                "directors of movies written",
                "directors of the films written",
                "directors of films written",
            ]):
                pattern = [
                    "written_by",
                    "written_by (reverse)",
                    "directed_by",
                ]

                if len(edges) == 3 and self._matches_prefix(edges, pattern):
                    return [(edges[2][0], edges[2][2])]

                return []

            if "starred" in q or "actors" in q:
                pattern = [
                    "starred_actors (reverse)",
                    "directed_by",
                ]

                if len(edges) == 2 and self._matches_prefix(edges, pattern):
                    return [(edges[1][0], edges[1][2])]

                return []

            final = edges[-1]

            if self._is_relation(final[1], target, reverse=False):
                return [(final[0], final[2])]

            return []

        if target == "written_by":
            final = edges[-1]

            if self._is_relation(final[1], target, reverse=False):
                return [(final[0], final[2])]

            return []

        if target == "starred_actors":
            if "director" in q or "directed" in q:
                pattern = [
                    "directed_by",
                    "directed_by (reverse)",
                    "starred_actors",
                ]

                if len(edges) == 3 and self._matches_prefix(edges, pattern):
                    return [(edges[2][0], edges[2][2])]

                return []

            final = edges[-1]

            if self._is_relation(final[1], target, reverse=False):
                return [(final[0], final[2])]

        return []

    def extract(self, query, evidence):
        target = self._target_relation(query)

        if target is None:
            return None

        answers = []
        sources = []
        q = query.casefold()

        is_co_director = any(x in q for x in [
            "co-director", "codirector", "co-directors",
        ])

        for item in evidence:
            if not isinstance(item, dict):
                continue

            path = item.get("path", [])

            if not isinstance(path, (list, tuple)) or len(path) < 2:
                continue

            if is_co_director:
                if len(path) != 3:
                    continue

                first, second = path[1], path[2]

                if (
                    not isinstance(first, (list, tuple))
                    or len(first) != 2
                    or not isinstance(second, (list, tuple))
                    or len(second) != 2
                ):
                    continue

                if (
                    str(first[0]).casefold() != "directed_by (reverse)"
                    or str(second[0]).casefold() != "directed_by"
                ):
                    continue

                candidates = [(2, str(second[1]).strip())]
            else:
                candidates = self._edge_candidates(query, path, target)

            item_answers = []

            for _, answer in candidates:
                answer = str(answer).strip()

                if not answer or answer.casefold() == "none":
                    continue

                if answer.casefold() not in {
                    value.casefold() for value in item_answers
                }:
                    item_answers.append(answer)

            for answer in item_answers:
                if answer.casefold() not in {
                    value.casefold() for value in answers
                }:
                    answers.append(answer)

            if item_answers:
                source = item.get("path_id")

                if source and source not in sources:
                    sources.append(source)

        if not answers:
            return None

        return {
            "answer": ", ".join(answers),
            "citations": sources,
            "method": "deterministic_kg_path_extraction",
        }
