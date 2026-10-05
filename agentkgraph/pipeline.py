from agentkgraph.config import Config
from agentkgraph.kg.graph_store import GraphStore
from agentkgraph.routing.router import Router
from agentkgraph.retrieval.kg_paths import KGRetriever
from agentkgraph.retrieval.vector import VectorRetriever
from agentkgraph.retrieval.hybrid import HybridRetriever


class Engine:
    def __init__(self,passages=None,config=None):
        self.config=config or Config()

        self.kg=GraphStore(
            self.config.kg
        )

        self.router=Router(
            self.config.routing
        )

        self.vector=VectorRetriever(
            passages or [],
            self.config.retrieval.top_k
        )

        self.kg_retriever=None
        self.hybrid=None

    def load_kg(self,path):
        self.kg.load(path)

        self.kg_retriever=KGRetriever(
            self.kg,
            self.config.kg.max_seeds
        )

        self.hybrid=HybridRetriever(
            self.vector,
            self.kg_retriever,
            self.config.retrieval.hybrid_alpha
        )

    def retrieve(self,query):
        if self.kg_retriever is None:
            raise RuntimeError(
                "KG not loaded. Call load_kg() first."
            )

        action=self.router.route(query)

        if action=="kg":
            evidence=self.kg_retriever.search(
                query
            )

        elif action=="vector":
            evidence=self.vector.search(
                query
            )

        else:
            evidence=self.hybrid.search(
                query
            )

        return {
            "query":query,
            "route":action,
            "evidence":evidence
        }

    def answer(self,query):
        result=self.retrieve(query)

        return {
            "query":query,
            "route":result["route"],
            "evidence":result["evidence"],
            "answer":self._dry_run_answer(
                result["evidence"]
            )
        }

    def _dry_run_answer(self,evidence):
        if not evidence:
            return "No sufficient evidence found."

        first=evidence[0]

        if isinstance(first,dict):
            if first.get("type")=="kg":
                first=first.get(
                    "evidence",
                    first
                )

            if "path" in first:
                path=first["path"]

                if path:
                    last=path[-1]

                    if isinstance(last,tuple):
                        return str(last[-1])

                    return str(last)

            if "answer" in first:
                return str(
                    first["answer"]
                )

            if "text" in first:
                return (
                    "Evidence retrieved; "
                    "synthesis required."
                )

        return "Evidence retrieved; synthesis required."