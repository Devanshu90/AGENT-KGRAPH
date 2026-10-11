
class KGPathAnswerExtractor:
    def __init__(self):
        self.relation_keywords = {
            "directed_by": ["directed", "director", "directors", "co-director", "codirector"],
            "starred_actors": ["starred", "stars", "actor", "actors", "cast", "appear"],
            "written_by": ["written", "writer", "writers"],
            "in_language": ["language", "languages"],
            "has_genre": ["genre", "genres"],
            "has_tags": ["tag", "tags"],
            "release_year": ["release year", "released"],
        }

    def _target_relation(self, query):
        q = query.lower()

        if any(word in q for word in ["language", "languages"]):
            return "in_language"

        if any(phrase in q for phrase in [
            "who starred", "who stars", "who acted",
            "who appears", "which actors", "what actors",
        ]):
            return "starred_actors"

        if any(word in q for word in [
            "co-director", "codirector", "co-directors",
            "directed", "director", "directors",
        ]):
            return "directed_by"

        if any(word in q for word in ["written", "writer", "writers"]):
            return "written_by"

        if any(word in q for word in ["genre", "genres"]):
            return "has_genre"

        if any(word in q for word in ["tag", "tags"]):
            return "has_tags"

        if any(word in q for word in ["release year", "released"]):
            return "release_year"

        if any(word in q for word in [
            "starred", "stars", "actor", "actors", "cast", "appear",
        ]):
            return "starred_actors"

        return None

    def _relations(self, path):
        return [
            str(item[0]).lower()
            for item in path
            if isinstance(item, (list, tuple))
            and len(item) == 2
            and item[0] is not None
        ]

    def _matches_pattern(self, query, relations, target):
        if not relations or target not in relations[-1]:
            return False

        q = query.lower()

        # Person -> movies they starred in -> directors.
        if target == "directed_by" and "starred" in q:
            return (
                len(relations) >= 2
                and "starred_actors" in relations[-2]
                and "reverse" in relations[-2]
                and "directed_by" in relations[-1]
                and "reverse" not in relations[-1]
            )

        # Director -> films directed by them -> other directors.
        if target == "directed_by" and any(
            word in q for word in ["co-director", "codirector", "co-directors"]
        ):
            return (
                len(relations) >= 2
                and "directed_by" in relations[-2]
                and "reverse" in relations[-2]
                and "directed_by" in relations[-1]
                and "reverse" not in relations[-1]
            )

        # Film -> director -> other films by the same director -> languages.
        if target == "in_language":
            return (
                len(relations) >= 3
                and "directed_by" in relations[0]
                and "directed_by" in relations[1]
                and "reverse" in relations[1]
                and "in_language" in relations[-1]
            )

        # Film -> director -> other films by the same director -> actors.
        if target == "starred_actors" and "director" in q:
            return (
                len(relations) >= 3
                and "directed_by" in relations[0]
                and "directed_by" in relations[1]
                and "reverse" in relations[1]
                and "starred_actors" in relations[-1]
                and "reverse" not in relations[-1]
            )

        # Questions asking which movies a named person appears in.
        # Starting from a person, the reverse edge leads to the movie.
        if target == "starred_actors":
            if "appear" in q and "in" in q:
                return (
                    "starred_actors" in relations[-1]
                    and "reverse" in relations[-1]
                )

            return (
                "starred_actors" in relations[-1]
                and "reverse" not in relations[-1]
            )

        return False

    def extract(self, query, evidence):
        target = self._target_relation(query)
        if target is None:
            return None

        answers = []
        sources = []

        for item in evidence:
            if not isinstance(item, dict):
                continue

            path = item.get("path", [])
            relations = self._relations(path)
            q = query.casefold()

            # For co-director questions, the answer may be an
            # intermediate director on a shared-movie path.
            is_co_director = any(
                word in q
                for word in ["co-director", "codirector", "co-directors"]
            )

            if is_co_director:
                if len(path) < 3 or len(relations) < 2:
                    continue

                first_relation = str(path[1][0]).casefold()
                answer_relation = str(path[2][0]).casefold()

                if not (
                    "directed_by" in first_relation
                    and "reverse" in first_relation
                    and "directed_by" in answer_relation
                    and "reverse" not in answer_relation
                ):
                    continue

                final_item = path[2]
            else:
                if not self._matches_pattern(query, relations, target):
                    continue
                final_item = path[-1]
            if not isinstance(final_item, (list, tuple)) or len(final_item) != 2:
                continue

            answer = str(final_item[1]).strip()
            if not answer:
                continue

            if answer.casefold() not in {x.casefold() for x in answers}:
                answers.append(answer)

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
