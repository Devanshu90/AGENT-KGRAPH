import json
import statistics
from pathlib import Path
from collections import Counter, defaultdict

d = Path("reports/b9_metaqa_100_qwen05b")
p = d / "b9_end_to_end_metaqa.json"
out = d / "b9_analysis.json"
txt = d / "b9_analysis.txt"

if not p.exists():
    raise SystemExit(f"Missing benchmark report: {p}")

data = json.loads(p.read_text(encoding="utf-8"))

def first(obj, names, default=None):
    if not isinstance(obj, dict):
        return default
    for name in names:
        if name in obj and obj[name] is not None:
            return obj[name]
    return default

results = data.get("results", [])
if not isinstance(results, list):
    raise SystemExit("Report's 'results' field is not a list.")

print("Report keys:", list(data.keys()))
print("Number of results:", len(results))
if results:
    print("Result fields:", list(results[0].keys()))

def num(r, keys):
    v = first(r, keys)
    try:
        return float(v) if v is not None else None
    except (ValueError, TypeError):
        return None

def hop_of(r):
    h = first(r, ["hop", "hop_count", "num_hops", "reasoning_hops"])
    if h is not None:
        try:
            return int(h)
        except (ValueError, TypeError):
            pass
    q = str(first(r, ["question", "query"], "")).lower()
    if "1-hop" in q or "1 hop" in q:
        return 1
    if "2-hop" in q or "2 hop" in q:
        return 2
    if "3-hop" in q or "3 hop" in q:
        return 3
    return None

groups = defaultdict(list)
for r in results:
    groups[hop_of(r)].append(r)

summary = {
    "source_report": str(p),
    "total_results": len(results),
    "hop_metrics": {},
    "route_counts": dict(Counter(
        str(first(r, ["route", "selected_route"], "unknown"))
        for r in results
    )),
    "failures": [],
    "notes": [
        "Failure categories are preliminary heuristics, not confirmed root causes.",
        "Evidence count alone cannot prove retrieval correctness or synthesis correctness."
    ]
}

lines = [
    "AGENT-KGRAPH B9 QWEN 0.5B ANALYSIS",
    "=" * 42,
    f"Results found: {len(results)}",
    "",
    "PERFORMANCE BY HOP",
    "-" * 42
]

for h in sorted(groups, key=lambda x: (x is None, x or 0)):
    rs = groups[h]
    ems = [num(r, ["em", "exact_match", "em_score"]) for r in rs]
    f1s = [num(r, ["f1", "token_f1", "token_f1_score"]) for r in rs]
    lats = [num(r, ["latency_ms", "latency", "latency_milliseconds"]) for r in rs]
    ems = [x for x in ems if x is not None]
    f1s = [x for x in f1s if x is not None]
    lats = [x for x in lats if x is not None]

    item = {
        "count": len(rs),
        "em_available": len(ems),
        "em_mean": statistics.mean(ems) if ems else None,
        "f1_available": len(f1s),
        "f1_mean": statistics.mean(f1s) if f1s else None,
        "latency_available": len(lats),
        "mean_latency_ms": statistics.mean(lats) if lats else None,
        "median_latency_ms": statistics.median(lats) if lats else None,
    }
    summary["hop_metrics"][str(h)] = item
    label = f"{h}-hop" if h is not None else "Unknown hop"
    lines.append(
        f"{label}: n={item['count']}, "
        f"EM={item['em_mean'] if item['em_mean'] is not None else 'N/A'}, "
        f"F1={item['f1_mean'] if item['f1_mean'] is not None else 'N/A'}, "
        f"mean latency={item['mean_latency_ms'] if item['mean_latency_ms'] is not None else 'N/A'} ms"
    )

lines += ["", "ROUTE COUNTS", "-" * 42]
for route, count in summary["route_counts"].items():
    lines.append(f"{route}: {count}")

lines += ["", "INCORRECT ANSWERS", "-" * 42]

for i, r in enumerate(results, 1):
    em = num(r, ["em", "exact_match", "em_score"])
    if em is None or em >= 1.0:
        continue

    question = first(r, ["question", "query"], "")
    answer = first(r, ["answer", "prediction", "predicted_answer"], "")
    gold = first(r, ["gold_answer", "expected_answer", "ground_truth", "target", "answers"], None)
    evidence = first(r, ["evidence", "evidence_count", "num_evidence"], None)
    route = first(r, ["route", "selected_route"], "unknown")
    f1 = num(r, ["f1", "token_f1", "token_f1_score"])

    if isinstance(evidence, list):
        evidence_count = len(evidence)
    else:
        try:
            evidence_count = int(evidence) if evidence is not None else None
        except (ValueError, TypeError):
            evidence_count = None

    if not str(answer).strip():
        category = "answer_extraction_or_synthesis"
        reason = "Prediction is empty; inspect extraction and synthesis stages."
    elif evidence_count == 0:
        category = "retrieval_or_missing_kg_facts"
        reason = "No evidence recorded; inspect retrieval and KG coverage."
    else:
        category = "needs_manual_attribution"
        reason = "Evidence exists; compare retrieved facts, expected answer, extraction, and synthesis."

    failure = {
        "result_index": i,
        "hop": hop_of(r),
        "question": question,
        "prediction": answer,
        "gold_answer": gold,
        "em": em,
        "f1": f1,
        "route": route,
        "evidence_count": evidence_count,
        "preliminary_category": category,
        "reason": reason
    }
    summary["failures"].append(failure)
    lines.append(
        f"\n#{i} | hop={hop_of(r)} | EM={em} | F1={f1} | "
        f"route={route} | evidence={evidence_count}"
    )
    lines.append(f"Q: {question}")
    lines.append(f"Predicted: {answer}")
    lines.append(f"Expected: {gold}")
    lines.append(f"Category: {category} — {reason}")

lines += [
    "",
    "LIMITATIONS",
    "-" * 42,
    "Root-cause labels are preliminary. A non-empty evidence list does not prove",
    "retrieval was correct; inspect actual evidence before attributing a failure.",
    "Check whether the report stores expected answers and evidence contents.",
    "A route count of KG for every record means this run does not compare routing arms."
]

out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("\nSaved analysis:")
print(out)
print(txt)
print("\nHop metrics:")
for h, m in summary["hop_metrics"].items():
    print(h, m)
print("\nRoute counts:", summary["route_counts"])
print("Incorrect predictions:", len(summary["failures"]))
print("\nFirst result example (for schema inspection):")
if results:
    print(json.dumps(results[0], ensure_ascii=False, indent=2)[:5000])
