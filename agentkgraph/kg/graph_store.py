import json
import networkx as nx
from dataclasses import dataclass


@dataclass
class Triple:
    subject:str
    predicate:str
    object:str
    passage_id:str
    confidence:float
    timestamp:float


class GraphStore:
    def __init__(self,config):
        self.config=config
        self.graph=nx.MultiDiGraph()

    def _kg_config(self):
        if hasattr(self.config,"kg"):
            return self.config.kg
        return self.config

    def add_entity(self,name,confidence=1.0):
        if name not in self.graph:
            self.graph.add_node(
                name,
                confidence=confidence
            )
        else:
            self.graph.nodes[name]["confidence"]=max(
                self.graph.nodes[name].get(
                    "confidence",
                    0.0
                ),
                confidence
            )

    def commit(self,triple):
        config=self._kg_config()

        if triple.confidence<config.commit_min_conf:
            return False

        self.add_entity(
            triple.subject,
            triple.confidence
        )

        self.add_entity(
            triple.object,
            triple.confidence
        )

        self.graph.add_edge(
            triple.subject,
            triple.object,
            predicate=triple.predicate,
            passage_id=triple.passage_id,
            confidence=triple.confidence,
            timestamp=triple.timestamp
        )

        return True

    def get_neighbors(self,entity):
        if entity not in self.graph:
            return []

        return list(
            self.graph.successors(entity)
        )

    def get_paths(self,entity,max_hops=None):
        config=self._kg_config()

        if max_hops is None:
            max_hops=config.hop_limit

        paths=[]

        if entity not in self.graph:
            return paths

        def dfs(node,path,score,depth):
            if depth>max_hops:
                return

            if len(path)>1:
                paths.append({
                    "path":path.copy(),
                    "score":score
                })

            if depth==max_hops:
                return

            for nxt in self.graph.successors(node):
                edges=self.graph.get_edge_data(
                    node,
                    nxt
                )

                if not edges:
                    continue

                for edge in edges.values():
                    node_conf=self.graph.nodes[nxt].get(
                        "confidence",
                        1.0
                    )

                    edge_conf=edge.get(
                        "confidence",
                        0.0
                    )

                    new_score=(
                        score*
                        node_conf*
                        edge_conf
                    )

                    if new_score<config.path_min_score:
                        continue

                    dfs(
                        nxt,
                        path+[
                            (
                                edge.get("predicate"),
                                nxt
                            )
                        ],
                        new_score,
                        depth+1
                    )

        dfs(
            entity,
            [(None,entity)],
            1.0,
            0
        )

        paths.sort(
            key=lambda x:x["score"],
            reverse=True
        )

        return paths[:config.top_paths]

    def decay(self):
        config=self._kg_config()
        remove_edges=[]

        for u,v,key,data in self.graph.edges(
            keys=True,
            data=True
        ):
            data["confidence"]*=(
                1-config.decay_rate
            )

            if (
                data["confidence"]<
                config.confidence_floor
            ):
                remove_edges.append(
                    (u,v,key)
                )

        self.graph.remove_edges_from(
            remove_edges
        )

    def save(self,path):
        data={
            "nodes":[],
            "edges":[]
        }

        for node,attrs in self.graph.nodes(
            data=True
        ):
            data["nodes"].append({
                "name":node,
                **attrs
            })

        for u,v,key,attrs in self.graph.edges(
            keys=True,
            data=True
        ):
            data["edges"].append({
                "subject":u,
                "object":v,
                "key":key,
                **attrs
            })

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                data,
                f,
                indent=2
            )

    def load(self,path):
        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:
            data=json.load(f)

        self.graph.clear()

        for node in data["nodes"]:
            node=dict(node)
            name=node.pop("name")

            self.graph.add_node(
                name,
                **node
            )

        for edge in data["edges"]:
            edge=dict(edge)

            subject=edge.pop("subject")
            obj=edge.pop("object")
            key=edge.pop("key",None)

            if key is None:
                self.graph.add_edge(
                    subject,
                    obj,
                    **edge
                )
            else:
                self.graph.add_edge(
                    subject,
                    obj,
                    key=key,
                    **edge
                )