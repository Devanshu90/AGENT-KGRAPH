
import json
from pathlib import Path

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
SAMPLE_SIZE = 30
FAILURES_PER_HOP = 10
PATHS_PER_FAILURE = 5
OUTPUT_PATH = "metaqa_diagnostic.json"


def normalize_answer(value):
    return " ".join(str(value or "").casefold().split())


def extract_answer_list(extractor, question, evidence):
    answers = []
    seen = set()

    for item in evidence:
        result = extractor.extract(question, [item])

        if not result:
            continue

        answer = result.get("answer", "").strip()
        key = normalize_answer(answer)

        if key and key not in seen:
            seen.add(key)
            answers.append(answer)

    return answers


want = {1: SAMPLE_SIZE, 2: SAMPLE_SIZE, 3: SAMPLE_SIZE}
cnt = {h: 0 for h in want}
rows = []

with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        h = int(r["hop"])

        if h in want and cnt[h] < want[h]:
            rows.append(r)
            cnt[h] += 1

        if all(cnt[h] == want[h] for h in want):
            break

cfg = Config()
cfg.kg.hop_limit = 3

eng = Engine(passages=[], config=cfg, dry_run=True)
eng.load_kg(GRAPH_PATH)
ext = KGPathAnswerExtractor()

stats = {
    h: {
        "total": 0,
        "em": 0,
        "f1": 0.0,
        "set_em": 0,
        "set_f1": 0.0,
        "empty": 0,
    }
    for h in want
}

failures = {h: [] for h in want}

try:
    for r in rows:
        h = int(r["hop"])
        question = r["question"]
        gold = r["answers"]

        result = eng.retrieve(question)
        evidence = result.get("evidence", [])

        out = ext.extract(question, evidence)
        pred = out["answer"] if out else ""

        pred_answers = extract_answer_list(
            ext,
            question,
            evidence,
        )

        em = exact_match(pred, gold)
        f1 = token_f1(pred, gold)
        sem = set_exact_match(pred_answers, gold)
        sf1 = set_token_f1(pred_answers, gold)

        s = stats[h]
        s["total"] += 1
        s["em"] += int(em)
        s["f1"] += f1
        s["set_em"] += int(sem)
        s["set_f1"] += sf1
        s["empty"] += int(not pred_answers)

        if not sem and len(failures[h]) < FAILURES_PER_HOP:
            failures[h].append({
                "question": question,
                "gold_answers": gold,
                "prediction": pred,
                "predicted_answers": pred_answers,
                "original_em": int(em),
                "original_f1": round(f1, 4),
                "set_em": int(sem),
                "set_f1": round(sf1, 4),
                "empty": not bool(pred_answers),
                "route": result.get("route"),
                "evidence_count": len(evidence),
                "paths": [
                    {
                        "path_id": e.get("path_id"),
                        "path": e.get("path"),
                        "score": e.get("score"),
                    }
                    for e in evidence[:PATHS_PER_FAILURE]
                ],
            })

    report = {
        "sample_counts": cnt,
        "by_hop": {},
        "incorrect_examples": {
            str(h): failures[h] for h in want
        },
        "failure_limit_per_hop": FAILURES_PER_HOP,
        "paths_shown_per_failure": PATHS_PER_FAILURE,
        "evaluation_notes": [
            "Original EM and F1 retain existing metric semantics.",
            "Set EM and set F1 compare individual predicted entities "
            "against the gold answer set.",
            "Individual predictions are extracted per evidence path; "
            "the combined answer string is not split on commas.",
            "Set F1 is entity-level set precision/recall F1, not token F1.",
        ],
    }

    for h, s in stats.items():
        total = s["total"]

        report["by_hop"][str(h)] = {
            **s,
            "em_rate": round(s["em"] / total, 4) if total else 0,
            "mean_f1": round(s["f1"] / total, 4) if total else 0,
            "set_em_rate": (
                round(s["set_em"] / total, 4) if total else 0
            ),
            "mean_set_f1": (
                round(s["set_f1"] / total, 4) if total else 0
            ),
        }

    Path(OUTPUT_PATH).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("MetaQA diagnostic complete")
    print(f"Detailed report: {OUTPUT_PATH}")
    print()

    for h in want:
        s = report["by_hop"][str(h)]

        print(
            f"{h}-hop: "
            f"Original EM {s['em']}/{s['total']} "
            f"({s['em_rate']:.1%}), "
            f"F1 {s['mean_f1']:.4f} | "
            f"Set EM {s['set_em']}/{s['total']} "
            f"({s['set_em_rate']:.1%}), "
            f"Set F1 {s['mean_set_f1']:.4f} | "
            f"empty {s['empty']}/{s['total']}"
        )

finally:
    eng.unload_all()
