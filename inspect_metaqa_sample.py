import json

seen = {1: 0, 2: 0, 3: 0}

with open("data/processed/metaqa/test.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        h = r["hop"]

        if h in seen and seen[h] < 2:
            print(json.dumps({
                "hop": h,
                "question": r["question"],
                "answers": r["answers"][:5]
            }, ensure_ascii=False))
            seen[h] += 1

        if all(v >= 2 for v in seen.values()):
            break
