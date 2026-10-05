class MetaQARetriever:
    def __init__(self,kg):
        self.kg=kg
        self.lookup={}

        for entity in kg.graph.nodes:
            self.lookup.setdefault(
                entity.lower().strip(),
                []
            ).append(entity)

    def resolve_entity(self,name):
        return self.lookup.get(
            name.lower().strip(),
            []
        )

    def neighbours(self,entity):
        results=[]

        for source,target,key,data in self.kg.graph.out_edges(
            entity,
            keys=True,
            data=True
        ):
            results.append((
                target,
                "out",
                data.get("predicate"),
                data.get("confidence",1.0)
            ))

        for source,target,key,data in self.kg.graph.in_edges(
            entity,
            keys=True,
            data=True
        ):
            results.append((
                source,
                "in",
                data.get("predicate"),
                data.get("confidence",1.0)
            ))

        return results

    def search(self,entity,hop):
        seeds=self.resolve_entity(entity)

        if not seeds:
            return []

        current=[
            (
                seed,
                [seed],
                1.0
            )
            for seed in seeds
        ]

        for _ in range(hop):
            nxt=[]

            for node,path,score in current:
                for target,direction,relation,confidence in self.neighbours(node):

                    if target in path:
                        continue

                    ns=score*confidence

                    nxt.append(
                        (
                            target,
                            path+[target],
                            ns
                        )
                    )

            current=nxt

            if not current:
                break

        current.sort(
            key=lambda x:x[2],
            reverse=True
        )

        seen=set()
        results=[]

        for node,path,score in current:
            key=node.lower().strip()

            if key in seen:
                continue

            seen.add(key)

            results.append({
                "answer":node,
                "path":path,
                "score":score
            })

        return results