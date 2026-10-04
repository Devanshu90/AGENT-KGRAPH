class KGRetriever:
    def __init__(self,kg,max_seeds=5):
        self.kg=kg
        self.max_seeds=max_seeds

    def find_entities(self,query):
        entities=[]

        q=query.lower()

        for entity in self.kg.graph.nodes:
            if entity.lower() in q:
                entities.append(entity)

        return entities[:self.max_seeds]

    def search(self,query):
        seeds=self.find_entities(query)

        results=[]

        for seed in seeds:
            for source,target,data in self.kg.graph.in_edges(
                seed,
                data=True
            ):
                score=(
                    self.kg.graph.nodes[source].get(
                        "confidence",1.0
                    )
                    *self.kg.graph.nodes[target].get(
                        "confidence",1.0
                    )
                    *data.get("confidence",0.0)
                )

                if score>=self.kg.config.path_min_score:
                    results.append({
                        "seed":seed,
                        "path":[
                            (None,source),
                            (data.get("predicate"),target)
                        ],
                        "score":score
                    })

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

        return results[:8]