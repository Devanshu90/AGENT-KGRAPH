import re
import math


class VectorRetriever:
    def __init__(self,top_k=4):
        self.top_k=top_k
        self.passages=[]

    def build(self,passages):
        self.passages=passages

    def _tokens(self,text):
        return set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                text.lower()
            )
        )

    def _score(self,query,text):
        q=self._tokens(query)
        t=self._tokens(text)

        if not q or not t:
            return 0.0

        overlap=len(q&t)

        return overlap/math.sqrt(
            len(q)*len(t)
        )

    def search(self,query):
        results=[]

        for passage in self.passages:
            score=self._score(
                query,
                passage["text"]
            )

            if score>0:
                results.append({
                    "passage_id":passage["passage_id"],
                    "text":passage["text"],
                    "score":score
                })

        results.sort(
            key=lambda x:x["score"],
            reverse=True
        )

        return results[:self.top_k]