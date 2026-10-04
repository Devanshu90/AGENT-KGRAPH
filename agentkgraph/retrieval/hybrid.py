class HybridRetriever:
    def __init__(self,vector_retriever,kg_retriever,alpha=0.5):
        self.vector=vector_retriever
        self.kg=kg_retriever
        self.alpha=alpha

    def search(self,query):
        vector_results=self.vector.search(query)
        kg_results=self.kg.search(query)

        return {
            "vector":vector_results,
            "kg":kg_results,
            "alpha":self.alpha
        }