import json
from agentkgraph.config import Config
from agentkgraph.pipeline import Engine
from agentkgraph.retrieval.answer_extractor import KGPathAnswerExtractor

p = "data/processed/metaqa/test.jsonl"
need = {1: 0, 2: 0, 3: 0}
rows = []

with open(p, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        h = int(r["hop"])
        if h in need and need[h] < 30:
            rows.append(r)
            need[h] += 1
        if all(v == 30 for v in need.values()):
            break

cfg = Config()
cfg.kg.hop_limit = 3
eng = Engine(passages=[], config=cfg, dry_run=True)
eng.load_kg("data/kg/metaqa_graph.json")
ext = KGPathAnswerExtractor()

try:
    for r in rows:
        ev = eng.retrieve(r["question"])["evidence"]
        out = ext.extract(r["question"], ev)
        if not out or not out.get("answer"):
            print(json.dumps({
                "hop": r["hop"],
                "question": r["question"],
                "gold": r["answers"],
                "path_count": len(ev),
                "paths": [
                    {"path": e.get("path"), "score": e.get("score")}
                    for e in ev[:3]
                ]
            }, ensure_ascii=False))
finally:
    eng.unload_all()
