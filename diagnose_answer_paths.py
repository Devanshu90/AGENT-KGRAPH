import json
from pathlib import Path

from agentkgraph.config import Config
from agentkgraph.pipeline import Engine
from agentkgraph.retrieval.answer_extractor import KGPathAnswerExtractor

rp = Path("reports/b9_metaqa_100_qwen05b/b9_end_to_end_metaqa.json")
if not rp.exists():
    raise FileNotFoundError(f"Benchmark report not found: {rp}")

data = json.loads(rp.read_text(encoding="utf-8"))
bad = [
    r for r in data["results"]
    if float(r.get("exact_match", 0)) < 1.0
]

print("Incorrect results:", len(bad))
print("Inspecting up to 6 representative failures")

cfg = Config()
cfg.kg.hop_limit = 3
cfg.models.synthesizer = "Qwen/Qwen2.5-0.5B-Instruct"
cfg.models.load_in_4bit = False

eng = Engine(passages=[], config=cfg, dry_run=False)
eng.load_kg("data/kg/metaqa_graph.json")

ext = KGPathAnswerExtractor()

terms = (
    "writer", "written", "screenwriter", "director",
    "genre", "language", "release year", "release years"
)

picked = []
seen = set()

for r in bad:
    q = r.get("question", "")
    if any(t in q.casefold() for t in terms):
        picked.append(r)
        seen.add(r.get("question_id"))
    if len(picked) >= 6:
        break

for r in bad:
    if len(picked) >= 6:
        break
    if r.get("question_id") not in seen:
        picked.append(r)
        seen.add(r.get("question_id"))

try:
    for r in picked:
        q = r["question"]
        evidence = eng.kg_retriever.search(q)
        target = ext._target_relation(q)
        extracted = ext.extract(q, evidence)

        print("\n" + "=" * 90)
        print("ID:", r.get("question_id"))
        print("HOP:", r.get("hop"))
        print("QUESTION:", q)
        print("EXPECTED:", r.get("answers"))
        print("BENCHMARK PREDICTION:", r.get("prediction"))
        print("EXTRACTOR TARGET:", target)
        print("EXTRACTOR ANSWER:", extracted["answer"] if extracted else None)
        print("PATH COUNT:", len(evidence))

        shown = 0
        for item in evidence:
            path = item.get("path", [])
            candidates = ext._edge_candidates(q, path, target) if target else []
            if candidates:
                print("PATH:", path)
                print("CANDIDATES:", candidates)
                shown += 1
                if shown >= 5:
                    break

        if shown == 0:
            print("No paths produced candidates for the selected target.")
finally:
    del eng

