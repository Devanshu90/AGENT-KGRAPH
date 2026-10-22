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


def load_rows():
    rows = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if int(row["hop"]) in (1, 2, 3):
                rows.append(row)
    return rows


def new_totals():
    return defaultdict(lambda: {
        "total": 0,
        "hit1": 0,
        "em": 0,
        "f1": 0.0,
        "set_em": 0,
        "set_f1": 0.0,
        "empty": 0,
    })


def restore_totals(details):
    totals = new_totals()

    for d in details:
        hop = int(d["hop"])
        s = totals[hop]
        s["total"] += 1
        s["hit1"] += int(d["hit1_proxy"])
        s["em"] += int(d["em"])
        s["f1"] += float(d["f1"])
        s["set_em"] += int(d["set_em"])
        s["set_f1"] += float(d["set_f1"])
        s["empty"] += int(not d.get("candidates"))

    return totals


def atomic_json_write(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def save_checkpoint(details, total):
    checkpoint = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed": len(details),
        "total": total,
        "details": details,
    }
    atomic_json_write(CHECKPOINT_PATH, checkpoint)


def load_checkpoint(total):
    if not os.path.exists(CHECKPOINT_PATH):
        print("No checkpoint found; starting from question 1.")
        return []

    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        checkpoint = json.load(f)

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint root must be a JSON object.")

    details = checkpoint.get("details")
    completed = checkpoint.get("completed")
    saved_total = checkpoint.get("total")

    if not isinstance(details, list):
        raise ValueError("Checkpoint details must be a list.")

    if saved_total != total:
        raise ValueError(
            f"Checkpoint total is {saved_total}, but current dataset "
            f"contains {total} questions. Refusing to resume."
        )

    if completed != len(details):
        raise ValueError(
            f"Checkpoint completed={completed}, but it contains "
            f"{len(details)} detail records."
        )

    if completed > total:
        raise ValueError("Checkpoint contains more results than the dataset.")

    for expected, item in enumerate(details, 1):
        if not isinstance(item, dict) or item.get("index") != expected:
            raise ValueError(
                f"Invalid checkpoint index at position {expected}. "
                "Expected contiguous indices starting at 1."
            )

    print(f"Validated checkpoint: {completed}/{total} questions.")
    return details


def build_summary(totals):
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

    return summary


def write_report(details, totals, elapsed):
    summary = build_summary(totals)

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
            "Full-split results should only be compared with paper results using matching metric definitions and evaluation protocols.",
            "F1 and set-F1 totals restored from a checkpoint use the six-decimal values saved per question, so tiny rounding differences from an uninterrupted run are possible."
        ],
        "by_hop": summary,
        "examples": details,
    }

    atomic_json_write(OUTPUT_PATH, report)

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
    print("Elapsed minutes this run:", round(elapsed / 60, 2))


def evaluate():
    start = time.time()
    rows = load_rows()
    total = len(rows)

    print("Loaded questions:", total)
    print("Counts:", {
        str(h): sum(int(r["hop"]) == h for r in rows)
        for h in (1, 2, 3)
    })

    details = load_checkpoint(total)
    completed = len(details)

    if completed == total:
        print("Checkpoint already contains every question.")
        totals = restore_totals(details)
        write_report(details, totals, time.time() - start)
        return

    totals = restore_totals(details)
    print(f"Resuming at question {completed + 1}/{total}.")

    cfg = Config()
    cfg.kg.hop_limit = 3
    eng = Engine(passages=[], config=cfg, dry_run=True)

    eng.load_kg(GRAPH_PATH)
    ext = KGPathAnswerExtractor()

    for i in range(completed + 1, total + 1):
        row = rows[i - 1]
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

        if i % PROGRESS_EVERY == 0 or i == total:
            elapsed = time.time() - start
            print(
                f"Progress: {i}/{total} ({i / total:.1%}), "
                f"this run elapsed={elapsed / 60:.1f} min"
            )
            save_checkpoint(details, total)

    write_report(details, totals, time.time() - start)


if __name__ == "__main__":
    evaluate()

