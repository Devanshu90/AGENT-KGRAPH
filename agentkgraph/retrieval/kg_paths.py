import re


class KGRetriever:
    def __init__(self,kg,max_seeds=5):
        self.kg=kg
        self.max_seeds=max_seeds
        self.entity_lookup={}

        for entity in self.kg.graph.nodes:
            key=entity.strip().lower()
            self.entity_lookup.setdefault(
                key,
                []
            ).append(entity)

    def find_entities(self,query):
        seeds=[]

        mentions=re.findall(
            r"\[([^\]]+)\]",
            query
        )

        if not mentions:
            mentions=[query]

        for mention in mentions:
            key=mention.strip().lower()

            for entity in self.entity_lookup.get(
                key,
                []
            ):
                if entity not in seeds:
                    seeds.append(entity)

            if len(seeds)>=self.max_seeds:
                break

        if not seeds:
            q=query.lower()

            matches=[]

            for key,entities in self.entity_lookup.items():
                if key in q:
                    for entity in entities:
                        matches.append(
                            (len(key),entity)
                        )

            matches.sort(
                key=lambda x:x[0],
                reverse=True
            )

            for _,entity in matches:
                if entity not in seeds:
                    seeds.append(entity)

                if len(seeds)>=self.max_seeds:
                    break

        return seeds

    def _edge_results(
        self,
        source,
        target,
        seed,
        data
    ):
        score=(
            self.kg.graph.nodes[source].get(
                "confidence",
                1.0
            )
            *self.kg.graph.nodes[target].get(
                "confidence",
                1.0
            )
            *data.get(
                "confidence",
                0.0
            )
        )

        if score<self.kg.config.path_min_score:
            return None

        return {
            "seed":seed,
            "path":[
                (None,source),
                (
                    data.get("predicate"),
                    target
                )
            ],
            "score":score
        }

    def search(self,query):
        seeds=self.find_entities(query)

        results=[]

        for seed in seeds:

            for source,target,key,data in self.kg.graph.in_edges(
                seed,
                keys=True,
                data=True
            ):
                result=self._edge_results(
                    source,
                    target,
                    seed,
                    data
                )

                if result is not None:
                    results.append(result)

            for source,target,key,data in self.kg.graph.out_edges(
                seed,
                keys=True,
                data=True
            ):
                result=self._edge_results(
                    source,
                    target,
                    seed,
                    data
                )

                if result is not None:
                    results.append(result)

            paths=self.kg.get_paths(seed)

            for path in paths:
                results.append({
                    "seed":seed,
                    "path":path["path"],
                    "score":path["score"]
                })

        results.sort(
            key=lambda x:x["score"],
            reverse=True
        )

        unique=[]
        seen=set()

        for result in results:
            key=(
                result["seed"],
                tuple(result["path"])
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(result)

        return unique[:8]