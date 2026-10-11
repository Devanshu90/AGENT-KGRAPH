import json
from agentkgraph.config import Config
from agentkgraph.pipeline import Engine
from agentkgraph.retrieval.answer_extractor import KGPathAnswerExtractor

path = "data/processed/metaqa/test.jsonl"
wanted = {1: 0, 2: 0, 3: 0}
records = []

with open(path, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        h = int(r["hop"])
        if h in wanted and wanted[h] < 2:
            records.append(r)
            wanted[h] += 1
        if all(v == 2 for v in wanted.values()):
            break

cfg = Config()
cfg.kg.hop_limit = 3
engine = Engine(passages=[], config=cfg, dry_run=True)
engine.load_kg("data/kg/metaqa_graph.json")
extractor = KGPathAnswerExtractor()

try:
    for r in records:
        q = r["question"]
        result = engine.retrieve(q)
        evidence = result["evidence"]
        extracted = extractor.extract(q, evidence)
        print(json.dumps({
            "hop": r["hop"],
            "question": q,
            "gold_answers": r["answers"],
            "route": result["route"],
            "evidence_count": len(evidence),
            "paths": [
                {
                    "path_id": e.get("path_id"),
                    "path": e.get("path"),
                    "score": e.get("score")
                }
                for e in evidence[:5]
            ],
            "extracted": extracted
        }, ensure_ascii=False))
finally:
    engine.unload_all()
