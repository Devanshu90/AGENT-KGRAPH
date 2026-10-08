import numpy as np
from sentence_transformers import SentenceTransformer


class VectorRetriever:
    def __init__(
        self,
        passages=None,
        top_k=4,
        model_name="BAAI/bge-small-en-v1.5",
        dry_run=False
    ):
        self.passages=passages or []
        self.top_k=top_k
        self.model_name=model_name
        self.dry_run=dry_run

        self.model=None
        self.embeddings=np.empty(
            (0,384),
            dtype=np.float32
        )

        if not self.dry_run:
            self._load_model()

    def _load_model(self):
        if self.model is not None:
            return

        self.model=SentenceTransformer(
            self.model_name
        )

        if self.passages:
            self._build_embeddings()

    def _build_embeddings(self):
        if self.model is None:
            self._load_model()

        texts=[
            str(p.get("text",""))
            for p in self.passages
        ]

        if not texts:
            self.embeddings=np.empty(
                (0,384),
                dtype=np.float32
            )
            return

        self.embeddings=self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        self.embeddings=np.asarray(
            self.embeddings,
            dtype=np.float32
        )

    def set_passages(self,passages):
        self.passages=passages or []

        if self.dry_run:
            self.embeddings=np.empty(
                (0,384),
                dtype=np.float32
            )
            return

        self._load_model()
        self._build_embeddings()

    def search(self,query):
        if not self.passages:
            return []

        if self.dry_run:
            return self._dry_search(
                query
            )

        if self.model is None:
            self._load_model()

        query_embedding=self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]

        query_embedding=np.asarray(
            query_embedding,
            dtype=np.float32
        )

        scores=np.dot(
            self.embeddings,
            query_embedding
        )

        indices=np.argsort(
            scores
        )[::-1][
            :self.top_k
        ]

        results=[]

        for i in indices:
            passage=self.passages[int(i)]

            results.append({
                "passage_id":passage.get(
                    "passage_id"
                ),
                "document_id":passage.get(
                    "document_id"
                ),
                "text":passage.get(
                    "text",
                    ""
                ),
                "score":float(
                    scores[int(i)]
                )
            })

        return results

    def _dry_search(self,query):
        q=set(
            str(query).lower().split()
        )

        scored=[]

        for passage in self.passages:
            text=str(
                passage.get(
                    "text",
                    ""
                )
            )

            words=set(
                text.lower().split()
            )

            score=(
                len(q & words)/
                max(len(q),1)
            )

            scored.append(
                (
                    score,
                    passage
                )
            )

        scored.sort(
            key=lambda x:x[0],
            reverse=True
        )

        results=[]

        for score,passage in scored[:self.top_k]:
            results.append({
                "passage_id":passage.get(
                    "passage_id"
                ),
                "document_id":passage.get(
                    "document_id"
                ),
                "text":passage.get(
                    "text",
                    ""
                ),
                "score":float(score)
            })

        return results

    def unload(self):
        if self.model is not None:
            del self.model
            self.model=None

        self.embeddings=np.empty(
            (0,384),
            dtype=np.float32
        )
