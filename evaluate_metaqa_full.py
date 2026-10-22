
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone

from agentkgraph.config import Config
from agentkgraph.pipeline import Engine
from agentkgraph.retrieval.answer_extractor import KGPathAnswerExtractor
from agentkgraph.evaluation.metrics import (
    exact_match,
    token_f1,
    set_exact_match,
    set_token_f1,
)

DATA_PATH = "data/processed/metaqa/test.jsonl"
GRAPH_PATH = "data/kg/metaqa_graph.json"
OUTPUT_PATH = "metaqa_full_evaluation.json"
CHECKPOINT_PATH = "metaqa_full_evaluation_checkpoint.json"
PROGRESS_EVERY = 250


def norm(value):
    return " ".join(str(value or "").casefold().split())


def as_list(value):
    if isinstance(value, list):
        return value
    return [value] if value else []


def evaluate():
    rows = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if int(row["hop"]) in (1, 2, 3):
                rows.append(row)

    print("Loaded questions:", len(rows))
    print("Counts:", {
        str(h): sum(int(r["hop"]) == h for r in rows)
        for h in (1, 2, 3)
    })

    cfg = Config()
    cfg.kg.hop_limit = 3
    eng = Engine(passages=[], config=cfg, dry_run=True)
    eng.load_kg(GRAPH_PATH)
    ext = KGPathAnswerExtractor()

    totals = defaultdict(lambda: {
        "total": 0, "hit1": 0, "em": 0, "f1": 0.0,
        "set_em": 0, "set_f1": 0.0, "empty": 0
    })
    details = []
    start = time.time()

    try:
        for i, row in enumerate(rows, 1):
            hop = int(row["hop"])
            question = row["question"]
            gold = as_list(row["answers"])

            result = eng.retrieve(question)
            evidence = result.get("evidence", [])

            combined = ext.extract(question, evidence)
            combined_answer = (
                str(combined.get("answer", "")).strip()
                if combined else ""
            )

            candidates = []
            seen = set()

            for rank, item in enumerate(evidence, 1):
                extracted = ext.extract(question, [item])
                if not extracted:
                    continue

                answer = str(extracted.get("answer", "")).strip()
                key = norm(answer)
                if not key or key in seen:
                    continue

                seen.add(key)
                candidates.append({
                    "candidate_rank_by_path_order": len(candidates) + 1,
                    "answer": answer,
                    "source_path_rank": rank,
                    "path_score": item.get("score"),
                    "path_id": item.get("path_id"),
                    "path": item.get("path"),
                })

            gold_norm = {norm(x) for x in gold if norm(x)}
            first_answer = candidates[0]["answer"] if candidates else ""

            # This is a proxy based on the first candidate from ranked paths.
            # It is not asserted to be an independently ranked answer model.
            hit1_proxy = bool(
                first_answer and norm(first_answer) in gold_norm
            )

            candidate_answers = [c["answer"] for c in candidates]
            em = exact_match(combined_answer, gold)
            f1 = token_f1(combined_answer, gold)
            sem = set_exact_match(candidate_answers, gold)
            sf1 = set_token_f1(candidate_answers, gold)

            s = totals[hop]
            s["total"] += 1
            s["hit1"] += int(hit1_proxy)
            s["em"] += int(em)
            s["f1"] += f1
            s["set_em"] += int(sem)
            s["set_f1"] += sf1
            s["empty"] += int(not candidates)

            details.append({
                "index": i,
                "hop": hop,
                "question": question,
                "gold_answers": gold,
                "combined_prediction": combined_answer,
                "first_candidate_proxy": first_answer,
                "hit1_proxy": int(hit1_proxy),
                "em": int(em),
                "f1": round(f1, 6),
                "set_em": int(sem),
                "set_f1": round(sf1, 6),
                "route": result.get("route"),
                "evidence_count": len(evidence),
                "candidates": candidates,
            })

            if i % PROGRESS_EVERY == 0 or i == len(rows):
                elapsed = time.time() - start
                print(
                    f"Progress: {i}/{len(rows)} "
                    f"({i / len(rows):.1%}), "
                    f"elapsed={elapsed / 60:.1f} min"
                )

                checkpoint = {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "completed": i,
                    "total": len(rows),
                    "details": details,
                }
                tmp = CHECKPOINT_PATH + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(checkpoint, f, ensure_ascii=False)
                os.replace(tmp, CHECKPOINT_PATH)

    finally:
        eng.close()

    summary = {}
    for hop in (1, 2, 3):
        s = totals[hop]
        n = s["total"]
        summary[str(hop)] = {
            **s,
            "hit1_proxy_rate": round(s["hit1"] / n, 6) if n else 0,
            "em_rate": round(s["em"] / n, 6) if n else 0,
            "mean_f1": round(s["f1"] / n, 6) if n else 0,
            "set_em_rate": round(s["set_em"] / n, 6) if n else 0,
            "mean_set_f1": round(s["set_f1"] / n, 6) if n else 0,
        }

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATA_PATH,
        "graph": GRAPH_PATH,
        "evaluation_size": len(details),
        "by_hop": summary,
        "hit1_definition": (
            "Proxy: first distinct answer extracted from evidence in the "
            "retriever's path order matches any gold answer after "
            "case-folding and whitespace normalization."
        ),
        "limitations": [
            "Path ranking is based on query/path relevance, not a separately trained answer ranking model.",
            "The first extracted answer is a proxy and must not be presented as verified paper-comparable Hit@1.",
            "This report uses the local processed dataset and KG; dataset version and preprocessing may differ from published benchmarks.",
            "Full-split results should only be compared with paper results using matching metric definitions and evaluation protocols."
        ],
        "by_hop": summary,
        "examples": details,
    }

    tmp = OUTPUT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)
    os.replace(tmp, OUTPUT_PATH)

    print("\nFULL EVALUATION SUMMARY")
    print("Hop | N | Hit@1 proxy | EM | Mean F1 | Set EM | Mean Set F1")
    for hop, s in summary.items():
        print(
            f"{hop:>3} | {s['total']:>5} | "
            f"{s['hit1_proxy_rate']:.2%} | "
            f"{s['em_rate']:.2%} | "
            f"{s['mean_f1']:.4f} | "
            f"{s['set_em_rate']:.2%} | "
            f"{s['mean_set_f1']:.4f}"
        )

    print("\nSaved:", OUTPUT_PATH)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Elapsed minutes:", round((time.time() - start) / 60, 2))


if __name__ == "__main__":
    evaluate()
