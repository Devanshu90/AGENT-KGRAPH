import json
from agentkgraph.config import Config
from agentkgraph.pipeline import Engine
from agentkgraph.retrieval.answer_extractor import KGPathAnswerExtractor

with open("metaqa_full_evaluation.json", encoding="utf-8") as f:
    report = json.load(f)

targets = {1: [], 2: [], 3: []}
for item in report["examples"]:
    hop = int(item["hop"])
    if hop in targets and not item.get("candidates") and len(targets[hop]) < 3:
        targets[hop].append(item)

cfg = Config()
cfg.kg.hop_limit = 3
eng = Engine(passages=[], config=cfg, dry_run=True)
eng.load_kg("data/kg/metaqa_graph.json")
ext = KGPathAnswerExtractor()

try:
    for hop, items in targets.items():
        for item in items:
            q = item["question"]
            result = eng.retrieve(q)
            ev = result.get("evidence", [])
            target = ext._target_relation(q)

            print("\n" + "=" * 80)
            print("HOP:", hop)
            print("QUESTION:", q)
            print("GOLD:", item.get("gold_answers"))
            print("TARGET RELATION:", target)
            print("EVIDENCE COUNT:", len(ev))

            for i, e in enumerate(ev[:8], 1):
                path = e.get("path", [])
                print("\nPATH", i, ":", path)
                print("SCORE:", e.get("score"))
                print("CANDIDATES:", ext._edge_candidates(q, path, target))

            print("\nFINAL EXTRACTION:", ext.extract(q, ev))
finally:
    eng.unload_all()
