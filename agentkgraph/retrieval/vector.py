from sentence_transformers import SentenceTransformer
import numpy as np


class VectorRetriever:
    def __init__(self,passages=None,top_k=4,model_name="BAAI/bge-small-en-v1.5"):
        self.passages=passages or []
        self.top_k=top_k
        self.model=SentenceTransformer(model_name)

        if self.passages:
            self.embeddings=self.model.encode(
                [p["text"] for p in self.passages],
                normalize_embeddings=True,
                show_progress_bar=True
            )
        else:
            self.embeddings=np.empty((0,384))

    def search(self,query):
        if not self.passages:
            return []

        query_embedding=self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        scores=np.dot(self.embeddings,query_embedding)
        indices=np.argsort(scores)[::-1][:self.top_k]

        results=[]

        for i in indices:
            results.append({
                "passage_id":self.passages[i]["passage_id"],
                "document_id":self.passages[i]["document_id"],
                "text":self.passages[i]["text"],
                "score":float(scores[i])
            })

        return results