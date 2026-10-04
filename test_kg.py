import time
from agentkgraph.config import Config
from agentkgraph.kg.graph_store import GraphStore,Triple


cfg=Config()
kg=GraphStore(cfg.kg)

kg.commit(
    Triple(
        "Christopher Nolan",
        "directed",
        "Inception",
        "p000001",
        0.95,
        time.time()
    )
)

kg.commit(
    Triple(
        "Inception",
        "released_in",
        "2010",
        "p000002",
        0.90,
        time.time()
    )
)

print("Nodes:",list(kg.graph.nodes))
print("Edges:",list(kg.graph.edges(data=True)))
print("Paths:",kg.get_paths("Christopher Nolan"))

kg.save("data/kg/graph.json")

print("KG saved successfully.")