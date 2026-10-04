import json

from agentkgraph.config import Config
from agentkgraph.kg.graph_store import GraphStore
from agentkgraph.retrieval.vector import VectorRetriever
from agentkgraph.retrieval.kg_paths import KGRetriever
from agentkgraph.retrieval.hybrid import HybridRetriever


def main():
    cfg=Config()

    kg=GraphStore(cfg.kg)
    kg.load("data/kg/graph.json")

    with open(
        "data/processed/kg_test.json",
        "r",
        encoding="utf-8"
    ) as f:
        passages=json.load(f)

    vr=VectorRetriever(
        cfg.retrieval.top_k
    )

    vr.build(passages)

    kr=KGRetriever(
        kg,
        cfg.kg.max_seeds
    )

    hr=HybridRetriever(
        vr,
        kr,
        cfg.retrieval.hybrid_alpha
    )

    query="Who directed Inception?"

    print("\nVECTOR RETRIEVAL")
    print(vr.search(query))

    print("\nKG RETRIEVAL")
    print(kr.search(query))

    print("\nHYBRID RETRIEVAL")
    print(hr.search(query))


if __name__=="__main__":
    main()