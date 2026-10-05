import json
import time
from pathlib import Path

from agentkgraph.config import Config
from agentkgraph.kg.graph_store import GraphStore,Triple


class MetaQAKGImporter:
    def __init__(self,config=None):
        self.config=config or Config()
        self.kg=GraphStore(self.config.kg)

    def canonicalize(self,name):
        return " ".join(str(name).strip().split())

    def import_file(self,input_path):
        input_path=Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"MetaQA KG file not found: {input_path}"
            )

        seen=set()
        total=0
        duplicates=0
        committed=0

        timestamp=time.time()

        with input_path.open(
            "r",
            encoding="utf-8"
        ) as f:
            for line in f:
                line=line.strip()

                if not line:
                    continue

                record=json.loads(line)

                subject=self.canonicalize(
                    record["subject"]
                )

                predicate=self.canonicalize(
                    record["predicate"]
                )

                obj=self.canonicalize(
                    record["object"]
                )

                key=(
                    subject,
                    predicate,
                    obj
                )

                total+=1

                if key in seen:
                    duplicates+=1
                    continue

                seen.add(key)

                triple=Triple(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    passage_id=(
                        f"metaqa:{record['triple_id']}"
                    ),
                    confidence=1.0,
                    timestamp=timestamp
                )

                if self.kg.commit(triple):
                    committed+=1

        return {
            "total":total,
            "unique":len(seen),
            "duplicates":duplicates,
            "committed":committed,
            "nodes":self.kg.graph.number_of_nodes(),
            "edges":self.kg.graph.number_of_edges()
        }

    def save(self,output_path):
        output_path=Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        self.kg.save(output_path)


def main():
    input_path=Path(
        "data/processed/metaqa/kg_triples.jsonl"
    )

    output_path=Path(
        "data/kg/metaqa_graph.json"
    )

    importer=MetaQAKGImporter()

    stats=importer.import_file(
        input_path
    )

    importer.save(
        output_path
    )

    print("="*60)
    print("METAQA KG IMPORT COMPLETE")
    print("="*60)
    print(f"Total triples:      {stats['total']}")
    print(f"Unique triples:     {stats['unique']}")
    print(f"Duplicates skipped: {stats['duplicates']}")
    print(f"Committed edges:    {stats['committed']}")
    print(f"Entities:           {stats['nodes']}")
    print(f"Graph edges:        {stats['edges']}")
    print(f"Output:             {output_path}")
    print("="*60)


if __name__=="__main__":
    main()