import re


class VectorRetriever:
    def __init__(self,passages=None,top_k=4):
        self.passages=passages or []
        self.top_k=top_k

    def _tokens(self,text):
        return set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                str(text).lower()
            )
        )

    def _score(self,query,text):
        q=self._tokens(query)
        p=self._tokens(text)

        if not q or not p:
            return 0.0

        return len(q&p)/len(q)

    def search(self,query):
        results=[]

        for passage in self.passages:
            score=self._score(
                query,
                passage["text"]
            )

            if score<=0:
                continue

            results.append({
                "passage_id":passage["passage_id"],
                "document_id":passage["document_id"],
                "text":passage["text"],
                "score":score
            })

        results.sort(
            key=lambda x:x["score"],
            reverse=True
        )

        return results[:self.top_k]