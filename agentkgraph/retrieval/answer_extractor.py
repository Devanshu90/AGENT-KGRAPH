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


    def _shared_kind(self, query):
        q = query.casefold()
        if any(x in q for x in [
            "share actor", "share actors", "shared actor", "shared actors",
            "common actor", "common actors", "same movie with", "same film with",
            "appeared in the same movie", "appeared in the same film",
            "acted in the same movie", "acted in the same film",
            "same actor of", "same actors of", "same actor as", "same actors as",
            "whose actors also appear in", "whose actors also appear",
            "actors also appear in", "actors who also appear in",
            "movies whose actors", "films whose actors",
        ]):
            return "starred_actors"
        if any(x in q for x in [
            "share director", "share directors", "shared director",
            "shared directors", "common director", "common directors",
        ]):
            return "directed_by"
        if any(x in q for x in [
            "share writer", "share writers", "shared writer",
            "shared writers", "common writer", "common writers",
        ]):
            return "written_by"
        return None


    def _target_relation(self, query):
        q = query.casefold()
        shared = self._shared_kind(query)
        if any(x in q for x in [
            "which languages", "what languages", "in which languages",
            "were in which languages", "primary languages",
            "what language", "which language",
            "languages spoken in", "language spoken in",
        ]):
            return "in_language"
        if any(x in q for x in [
            "which genres", "what genres", "which genre",
            "what genre", "fall under", "types of",
            "what types are", "what types", "genres of", "genre of",
        ]):
            return "has_genre"
        if any(x in q for x in [
            "release year", "release years", "what year",
            "which year", "when were", "when did",
        ]):
            return "release_year"
        if "tag" in q:
            return "has_tags"
        if shared:
            if any(x in q for x in [
                "director of", "directors of", "director for",
                "directors for", "who directed", "which director",
                "what director", "who is listed as director",
                "which directors", "who was the director",
                "are directed by", "were directed by",
                "was directed by",
            ]):
                return "directed_by"
            if any(x in q for x in [
                "who wrote", "which writer", "what writer",
                "written by", "screenwriter",
            ]):
                return "written_by"
            return "shared_movies"
        if any(x in q for x in [
            "who starred", "who stars", "who acted",
            "who appears", "which actors", "what actors",
            "who are the actors", "which actor",
            "who appeared in the same movie", "who appeared in the same film",
            "who acted in the same movie", "who acted in the same film",
        ]):
            return "starred_actors"
        if any(x in q for x in [
            "films did", "what films", "which movies", "which films",
            "movies did", "movies was", "appear in",
            "what does", "appears in which movies",
            "starred in which movies", "what movies",
            "also directed which movies", "also directed what movies",
        ]):
            if "director" in q or "directed" in q:
                return "filmography_director"
            if "writer" in q or "written" in q:
                return "filmography_writer"
            return "filmography"
        if any(x in q for x in [
            "director of", "directors of", "director for",
            "directors for", "who directed", "which director",
            "what director", "who is listed as director",
            "which directors", "who was the director",
            "co-director", "codirector",
        ]):
            if "writer of" in q and (
                "movies for" in q or "films for" in q
            ):
                return "written_by"
            return "directed_by"
        if any(x in q for x in [
            "who wrote", "which writer", "what writer",
            "written by", "screenwriter", "writer of",
        ]):
            return "written_by"
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


    def _is_relation(self, relation, target, reverse=None):
        relation = str(relation).casefold()
        if target not in relation:
            return False
        is_reverse = "reverse" in relation
        return reverse is None or is_reverse == reverse


    def _matches_prefix(self, edges, relations):
        if len(edges) < len(relations):
            return False
        return all(
            actual == expected
            for (_, actual, _), expected in zip(edges, relations)
        )


    def _shared_prefixes(self, kind):
        return [
            [kind, kind + " (reverse)"],
            [kind + " (reverse)", kind],
        ]


    def _edge_candidates(self, query, path, target):
        q = query.casefold()
        edges = self._edges(path)
        if not edges:
            return []
        shared = self._shared_kind(query)
        if target in {"in_language", "has_genre", "release_year", "has_tags"}:
            if shared:
                for prefix in self._shared_prefixes(shared):
                    if len(edges) >= 3 and self._matches_prefix(edges, prefix):
                        final = edges[2]
                        if self._is_relation(final[1], target, reverse=False):
                            return [(final[0], final[2])]
                return []
            for idx, rel, value in edges:
                if self._is_relation(rel, target, reverse=False):
                    return [(idx, value)]
            return []
        if target == "filmography":
            idx, rel, value = edges[0]
            if self._is_relation(rel, "starred_actors", reverse=True):
                return [(idx, value)]
            seed = str(path[0][1]).strip()
            if (
                self._is_relation(rel, "starred_actors", reverse=False)
                and value.casefold() == seed.casefold()
            ):
                return [(idx, value)]
            return []
        if target == "filmography_director":
            if self._matches_prefix(
                edges, ["directed_by", "directed_by (reverse)"]
            ):
                return [(edges[1][0], edges[1][2])]
            return []
        if target == "filmography_writer":
            if self._matches_prefix(
                edges, ["written_by", "written_by (reverse)"]
            ):
                return [(edges[1][0], edges[1][2])]
            return []
        if target == "shared_movies":
            if shared is None or len(edges) < 2:
                return []
            for prefix in self._shared_prefixes(shared):
                if self._matches_prefix(edges, prefix):
                    return [(edges[1][0], edges[1][2])]
            return []
        if target == "directed_by":
            if "writer of" in q and (
                "movies written by" in q or "films written by" in q
            ):
                pattern = [
                    "written_by",
                    "written_by (reverse)",
                    "directed_by",
                ]
                if len(edges) >= 3 and self._matches_prefix(edges, pattern):
                    return [(edges[2][0], edges[2][2])]
                return []
            if shared == "written_by":
                pattern = [
                    "written_by",
                    "written_by (reverse)",
                    "directed_by",
                ]
                if len(edges) >= 3 and self._matches_prefix(edges, pattern):
                    return [(edges[2][0], edges[2][2])]
                return []
            if shared == "starred_actors":
                patterns = [
                    [
                        "starred_actors",
                        "starred_actors (reverse)",
                        "directed_by",
                    ],
                    [
                        "starred_actors (reverse)",
                        "starred_actors",
                        "directed_by",
                    ],
                ]
                for pattern in patterns:
                    if len(edges) >= 3 and self._matches_prefix(edges, pattern):
                        return [(edges[2][0], edges[2][2])]
                return []
            if self._matches_prefix(
                edges, ["directed_by", "directed_by (reverse)"]
            ):
                return [(edges[1][0], edges[1][2])]
            for idx, rel, value in edges:
                if self._is_relation(rel, "directed_by", reverse=False):
                    return [(idx, value)]
            return []
        if target == "written_by":
            for idx, rel, value in edges:
                if self._is_relation(rel, "written_by", reverse=False):
                    return [(idx, value)]
            return []
        if target == "starred_actors":
            if "director" in q or "directed" in q:
                if (
                    len(edges) == 1
                    and self._is_relation(
                        edges[0][1], "starred_actors", reverse=False
                    )
                ):
                    return [(edges[0][0], edges[0][2])]
                patterns = [
                    [
                        "directed_by",
                        "directed_by (reverse)",
                        "starred_actors",
                    ],
                    [
                        "directed_by (reverse)",
                        "starred_actors",
                    ],
                ]
                for pattern in patterns:
                    if len(edges) >= len(pattern) and self._matches_prefix(edges, pattern):
                        idx = len(pattern) - 1
                        return [(edges[idx][0], edges[idx][2])]
                return []
            for idx, rel, value in edges:
                if self._is_relation(rel, "starred_actors", reverse=False):
                    return [(idx, value)]
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
                edges = self._edges(path)
                candidates = []
                if len(edges) >= 2 and self._matches_prefix(
                    edges, ["directed_by (reverse)", "directed_by"]
                ):
                    candidates = [(edges[1][0], edges[1][2])]
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
