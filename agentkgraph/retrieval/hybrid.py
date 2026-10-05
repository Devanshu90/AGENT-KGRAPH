class HybridRetriever:
    def __init__(
        self,
        vector_retriever,
        kg_retriever,
        alpha=0.5
    ):
        self.vector=vector_retriever
        self.kg=kg_retriever
        self.alpha=alpha

    def search(self,query):
        vector_results=self.vector.search(query)
        kg_results=self.kg.search(query)

        results=[]

        for item in vector_results:
            results.append({
                "type":"vector",
                "score":(
                    self.alpha
                    *item["score"]
                ),
                "evidence":item
            })

        for item in kg_results:
            results.append({
                "type":"kg",
                "score":(
                    (1-self.alpha)
                    *item["score"]
                ),
                "evidence":item
            })

        results.sort(
            key=lambda x:x["score"],
            reverse=True
        )

        return results