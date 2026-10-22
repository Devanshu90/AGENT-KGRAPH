
import json
from collections import defaultdict

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
OUTPUT_PATH = "metaqa_hit1_results.json"


def norm(x):
    return " ".join(str(x or "").casefold().split())


def as_list(x):
    if isinstance(x, list):
        return x
    return [x] if x else []


def main():
    counts = {1: 0, 2: 0, 3: 0}
    rows = []

    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            h = int(r["hop"])
            if h in counts and counts[h] < SAMPLE_SIZE:
                rows.append(r)
                counts[h] += 1
            if all(v == SAMPLE_SIZE for v in counts.values()):
                break

    if any(v == 0 for v in counts.values()):
        raise RuntimeError(f"Missing hop samples: {counts}")

    cfg = Config()
    cfg.kg.hop_limit = 3
    eng = Engine(passages=[], config=cfg, dry_run=True)
    eng.load_kg(GRAPH_PATH)
    ext = KGPathAnswerExtractor()

    stats = {
        h: {"total": 0, "hit1": 0, "em": 0, "f1": 0.0,
            "set_em": 0, "set_f1": 0.0, "empty": 0}
        for h in counts
    }
    details = []

    for r in rows:
        h = int(r["hop"])
        q = r["question"]
        gold = as_list(r["answers"])
        result = eng.retrieve(q)
        evidence = result.get("evidence", [])

        out = ext.extract(q, evidence)
        pred = out.get("answer", "") if out else ""

        candidates = []
        seen = set()
        for e in evidence:
            one = ext.extract(q, [e])
            if not one:
                continue
            ans = str(one.get("answer", "")).strip()
            key = norm(ans)
            if key and key not in seen:
                seen.add(key)
                candidates.append(ans)

        gold_norm = {norm(x) for x in gold if norm(x)}
        top_prediction = candidates[0] if candidates else ""
        hit = bool(top_prediction and norm(top_prediction) in gold_norm)

        em = exact_match(pred, gold)
        f1 = token_f1(pred, gold)
        sem = set_exact_match(candidates, gold)
        sf1 = set_token_f1(candidates, gold)

        s = stats[h]
        s["total"] += 1
        s["hit1"] += int(hit)
        s["em"] += int(em)
        s["f1"] += f1
        s["set_em"] += int(sem)
        s["set_f1"] += sf1
        s["empty"] += int(not candidates)

        details.append({
            "hop": h,
            "question": q,
            "gold_answers": gold,
            "combined_prediction": pred,
            "ordered_candidates": candidates,
            "hit_at_1": int(hit),
            "em": int(em),
            "f1": round(f1, 4),
            "set_em": int(sem),
            "set_f1": round(sf1, 4),
            "route": result.get("route"),
            "evidence_count": len(evidence),
        })

    summary = {}
    for h, s in stats.items():
        n = s["total"]
        summary[str(h)] = {
            **s,
            "hit1_rate": round(s["hit1"] / n, 4) if n else 0,
            "em_rate": round(s["em"] / n, 4) if n else 0,
            "mean_f1": round(s["f1"] / n, 4) if n else 0,
            "set_em_rate": round(s["set_em"] / n, 4) if n else 0,
            "mean_set_f1": round(s["set_f1"] / n, 4) if n else 0,
        }

    report = {
        "sample_counts": counts,
        "by_hop": summary,
        "hit1_definition": (
            "Top candidate in evidence iteration order exactly matches "
            "a gold answer after case-folding and whitespace normalization."
        ),
        "warning": (
            "This is a 30-question-per-hop diagnostic sample. Candidate "
            "iteration order has not been verified as a calibrated ranking; "
            "Hit@1 is provisional and not directly comparable to paper results."
        ),
        "examples": details,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("MetaQA diagnostic evaluation")
    print("Hop | N | Hit@1* | EM | Mean F1 | Set EM | Mean Set F1")
    for h, s in summary.items():
        print(
            f"{h:>3} | {s['total']:>2} | "
            f"{s['hit1_rate']:.1%} | {s['em_rate']:.1%} | "
            f"{s['mean_f1']:.4f} | {s['set_em_rate']:.1%} | "
            f"{s['mean_set_f1']:.4f}"
        )
    print("\nSaved:", OUTPUT_PATH)
    print("*Provisional: first extracted candidate, not verified model ranking.")


if __name__ == "__main__":
    main()
