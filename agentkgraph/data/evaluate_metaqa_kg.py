import json
import re
from pathlib import Path

from agentkgraph.config import Config
from agentkgraph.kg.graph_store import GraphStore
from agentkgraph.data.metaqa_retriever import MetaQARetriever


def normalize(text):
    text=str(text).lower().strip()
    text=re.sub(r"\s+"," ",text)
    return text


def extract_entity(question):
    m=re.search(
        r"\[([^\]]+)\]",
        question
    )

    if m:
        return m.group(1).strip()

    return None


def evaluate_split(split,max_questions=None):
    kg=GraphStore(Config().kg)
    kg.load("data/kg/metaqa_graph.json")

    retriever=MetaQARetriever(kg)

    path=Path(
        f"data/processed/metaqa/{split}.jsonl"
    )

    total=0
    hit=0
    full=0

    hop_stats={
        1:{"total":0,"hit":0},
        2:{"total":0,"hit":0},
        3:{"total":0,"hit":0}
    }

    with path.open(
        encoding="utf-8"
    ) as f:

        for line in f:
            record=json.loads(line)

            entity=extract_entity(
                record["question"]
            )

            if entity is None:
                continue

            answers={
                normalize(x)
                for x in record["answers"]
            }

            results=retriever.search(
                entity,
                record["hop"]
            )

            candidates={
                normalize(x["answer"])
                for x in results
            }

            is_hit=bool(
                answers & candidates
            )

            is_full=answers.issubset(
                candidates
            )

            total+=1

            if is_hit:
                hit+=1

            if is_full:
                full+=1

            hop=record["hop"]

            if hop in hop_stats:
                hop_stats[hop]["total"]+=1

                if is_hit:
                    hop_stats[hop]["hit"]+=1

            if (
                max_questions is not None
                and total>=max_questions
            ):
                break

    return {
        "total":total,
        "hit":hit,
        "hit_rate":hit/total if total else 0,
        "full":full,
        "full_rate":full/total if total else 0,
        "hop_stats":hop_stats
    }


def main():
    print("="*60)
    print("METAQA GENERIC KG BASELINE")
    print("="*60)

    for split in [
        "train",
        "dev",
        "test"
    ]:
        stats=evaluate_split(split)

        print()
        print(split.upper())
        print(
            "Questions:",
            stats["total"]
        )
        print(
            "Answer hit:",
            stats["hit"]
        )
        print(
            "Hit rate:",
            f"{stats['hit_rate']:.4f}"
        )
        print(
            "Full answer rate:",
            f"{stats['full_rate']:.4f}"
        )

        for hop,data in stats[
            "hop_stats"
        ].items():

            if data["total"]:
                rate=(
                    data["hit"]
                    /data["total"]
                )

                print(
                    f"{hop}-hop:",
                    data["hit"],
                    "/",
                    data["total"],
                    f"({rate:.4f})"
                )


if __name__=="__main__":
    main()