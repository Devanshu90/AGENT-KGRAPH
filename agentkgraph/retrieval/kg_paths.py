
import re
import unicodedata


class KGRetriever:
    def __init__(self, kg, max_seeds=5):
        self.kg = kg
        self.max_seeds = max_seeds
        self.entity_lookup = {}

        for entity in self.kg.graph.nodes:
            key = self._normalize_entity(entity)
            self.entity_lookup.setdefault(key, []).append(entity)

    def _normalize_entity(self, value):
        value = unicodedata.normalize("NFKD", str(value))
        value = "".join(
            c for c in value
            if not unicodedata.combining(c)
        )
        return re.sub(r"\s+", " ", value).strip().casefold()

    def find_entities(self, query):
        seeds = []
        mentions = re.findall(r"\[([^\]]+)\]", query)

        for mention in mentions:
            key = self._normalize_entity(mention)

            for entity in self.entity_lookup.get(key, []):
                if not any(
                    self._normalize_entity(existing)
                    == self._normalize_entity(entity)
                    for existing in seeds
                ):
                    seeds.append(entity)

            if not any(
                self._normalize_entity(entity) == key
                for entity in seeds
            ):
                mention_tokens = set(key.split())
                candidates = []

                for entities in self.entity_lookup.values():
                    for entity in entities:
                        entity_key = self._normalize_entity(entity)
                        entity_tokens = set(entity_key.split())
                        overlap = mention_tokens & entity_tokens

                        if overlap:
                            score = (
                                len(overlap),
                                len(overlap) / max(len(entity_tokens), 1)
                            )
                            candidates.append((score, entity))

                if candidates:
                    candidates.sort(
                        key=lambda item: item[0],
                        reverse=True
                    )
                    best_score = candidates[0][0]
                    best = [
                        entity
                        for score, entity in candidates
                        if score == best_score
                    ]

                    if len(best) == 1 and best[0] not in seeds:
                        seeds.append(best[0])

            if len(seeds) >= self.max_seeds:
                break

        # Avoid accidental entity matches when an explicit entity
        # mention is already present in the question.
        if not seeds and not mentions:
            query_normalized = self._normalize_entity(query)
            matches = []

            for key, entities in self.entity_lookup.items():
                if len(key) >= 4 and re.search(
                    r"(?<!\w)" + re.escape(key) + r"(?!\w)",
                    query_normalized
                ):
                    for entity in entities:
                        matches.append((len(key), entity))

            matches.sort(
                key=lambda item: item[0],
                reverse=True
            )

            for _, entity in matches:
                if entity not in seeds:
                    seeds.append(entity)

                if len(seeds) >= self.max_seeds:
                    break

        return seeds

    def _edge_score(self, source, target, data):
        return (
            self.kg.graph.nodes[source].get("confidence", 1.0)
            * self.kg.graph.nodes[target].get("confidence", 1.0)
            * data.get("confidence", 0.0)
        )

    def _edge_results(self, source, target, seed, data):
        score = self._edge_score(source, target, data)

        if score < self.kg.config.path_min_score:
            return None

        predicate = data.get("predicate", "related_to")

        return {
            "seed": seed,
            "path": [source, predicate, target],
            "score": score
        }

    def _multi_hop_paths(self, seed):
        max_hops = self.kg.config.hop_limit
        min_score = self.kg.config.path_min_score
        results = []

        def dfs(node, path, score, depth, visited):
            if depth >= max_hops:
                return

            edges = []

            for source, target, key, data in (
                self.kg.graph.out_edges(
                    node,
                    keys=True,
                    data=True
                )
            ):
                edges.append((source, target, data, False))

            for source, target, key, data in (
                self.kg.graph.in_edges(
                    node,
                    keys=True,
                    data=True
                )
            ):
                edges.append((source, target, data, True))

            for source, target, data, reversed_edge in edges:
                next_node = source if reversed_edge else target

                if next_node in visited:
                    continue

                edge_score = self._edge_score(
                    source,
                    target,
                    data
                )
                new_score = score * edge_score

                if new_score < min_score:
                    continue

                predicate = data.get("predicate", "related_to")

                if reversed_edge:
                    relation = f"{predicate} (reverse)"
                else:
                    relation = predicate

                new_path = path + [(relation, next_node)]

                results.append({
                    "seed": seed,
                    "path": new_path,
                    "score": new_score
                })

                dfs(
                    next_node,
                    new_path,
                    new_score,
                    depth + 1,
                    visited | {next_node}
                )

        dfs(
            seed,
            [(None, seed)],
            1.0,
            0,
            {seed}
        )

        return results

    def _relevant_relations(self, query):
        query_lower = query.lower()

        mapping = {
            "starred_actors": [
                "starred",
                "star",
                "appears",
                "appear",
                "acted",
                "actor",
                "actors",
                "cast",
                "starred_actors"
            ],
            "directed_by": [
                "directed",
                "director",
                "directors",
                "direct",
                "co-director",
                "codirector",
                "co-directors"
            ],
            "written_by": [
                "written",
                "writer",
                "writers"
            ],
            "in_language": [
                "language",
                "languages"
            ],
            "has_genre": [
                "genre",
                "genres"
            ],
            "has_tags": [
                "tag",
                "tags"
            ],
            "release_year": [
                "release year",
                "released",
                "year"
            ]
        }

        relevant = set()

        for relation, words in mapping.items():
            for word in words:
                if word in query_lower:
                    relevant.add(relation)
                    break

        return relevant

    def _target_relations(self, query):
        query_lower = query.lower()

        # Prioritize the requested answer type over intermediate relations.
        if any(
            word in query_lower
            for word in ["language", "languages"]
        ):
            return ["in_language"]

        if any(
            phrase in query_lower
            for phrase in [
                "who starred",
                "who stars",
                "who acted",
                "who appears",
                "which actors",
                "what actors"
            ]
        ):
            return ["starred_actors"]

        if any(
            word in query_lower
            for word in [
                "directed",
                "director",
                "directors",
                "co-director",
                "codirector",
                "co-directors"
            ]
        ):
            return ["directed_by"]

        if any(
            word in query_lower
            for word in ["written", "writer", "writers"]
        ):
            return ["written_by"]

        if any(
            word in query_lower
            for word in ["genre", "genres"]
        ):
            return ["has_genre"]

        if any(
            word in query_lower
            for word in ["tag", "tags"]
        ):
            return ["has_tags"]

        if any(
            word in query_lower
            for word in ["release year", "released"]
        ):
            return ["release_year"]

        if any(
            word in query_lower
            for word in [
                "starred",
                "stars",
                "actor",
                "actors",
                "cast",
                "appear"
            ]
        ):
            return ["starred_actors"]

        return []

    def _path_relation_score(self, path, relevant_relations):
        if not relevant_relations:
            return 0.0

        score = 0.0

        for item in path:
            if not isinstance(item, (list, tuple)):
                continue

            if len(item) != 2:
                continue

            relation = str(item[0]).lower()

            for target in relevant_relations:
                if target.lower() in relation:
                    score += 3.0

        return score

    def _final_relation_score(self, path, target_relations):
        if not target_relations or len(path) < 2:
            return 0.0

        final_relation = str(path[-1][0]).lower()

        for relation in target_relations:
            if relation.lower() in final_relation:
                return 10.0

        return 0.0

    def _rank_path(
        self,
        result,
        relevant_relations,
        target_relations
    ):
        path = result["path"]
        query = getattr(self, "_active_query", "").casefold()

        # Prefer direct shared-film paths for co-director questions.
        if any(
            word in query
            for word in [
                "co-director",
                "codirector",
                "co-directors"
            ]
        ):
            relations = [
                str(item[0]).casefold()
                for item in path
                if isinstance(item, (list, tuple))
                and len(item) == 2
                and item[0] is not None
            ]

            if (
                len(path) == 3
                and len(relations) == 2
                and "directed_by" in relations[0]
                and "reverse" in relations[0]
                and "directed_by" in relations[1]
                and "reverse" not in relations[1]
            ):
                return 100.0 + result["score"] * 0.01

        relation_score = self._path_relation_score(
            path,
            relevant_relations
        )

        final_relation_score = self._final_relation_score(
            path,
            target_relations
        )

        hop_count = max(len(path) - 1, 0)
        target_bonus = (
            2.0
            if final_relation_score > 0 and hop_count >= 2
            else 0.0
        )

        return (
            final_relation_score
            + relation_score
            + target_bonus
            + result["score"] * 0.01
        )

    def search(self, query):
        self._active_query = query
        seeds = self.find_entities(query)
        results = []

        for seed in seeds:
            results.extend(
                self._multi_hop_paths(seed)
            )

        relevant_relations = self._relevant_relations(query)
        target_relations = self._target_relations(query)

        results.sort(
            key=lambda item: self._rank_path(
                item,
                relevant_relations,
                target_relations
            ),
            reverse=True
        )

        unique = []
        seen = set()

        for result in results:
            path_key = tuple(
                str(item)
                for item in result["path"]
            )

            if path_key in seen:
                continue

            seen.add(path_key)
            result["path_id"] = (
                f"kg_path_{len(unique) + 1:04d}"
            )
            unique.append(result)

        return unique[:50]
