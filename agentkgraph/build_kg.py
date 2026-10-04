import time
from pathlib import Path

from agentkgraph.config import Config
from agentkgraph.kg.graph_store import GraphStore,Triple
from agentkgraph.agents.extractor import Extractor
from agentkgraph.agents.verifier import Verifier
from agentkgraph.agents.merger import EntityMerger


class KGBuildPipeline:
    def __init__(self,config=None):
        self.config=config or Config()

        self.kg=GraphStore(self.config.kg)

        self.extractor=Extractor(dry_run=True)

        self.verifier=Verifier(
            threshold=0.60,
            dry_run=True
        )

        self.merger=EntityMerger(
            threshold=0.92
        )

        self.entities=set()

    def process_passage(self,text,passage_id):
        extracted=self.extractor.extract(
            text,
            passage_id
        )

        committed=0

        for candidate in extracted:
            verification=self.verifier.verify(
                candidate,
                text
            )

            if not verification.accepted:
                continue

            candidate.confidence=min(
                candidate.confidence,
                verification.confidence
            )

            candidate=self.merger.merge_triple(
                candidate,
                self.entities
            )

            triple=Triple(
                subject=candidate.subject,
                predicate=candidate.predicate,
                object=candidate.object,
                passage_id=candidate.passage_id,
                confidence=candidate.confidence,
                timestamp=time.time()
            )

            if self.kg.commit(triple):
                committed+=1

        return committed

    def build(self,passages):
        total=0

        for passage in passages:
            total+=self.process_passage(
                passage["text"],
                passage["passage_id"]
            )

        return total

    def save(self,path):
        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.kg.save(path)