import json

with open("metaqa_diagnostic.json", encoding="utf-8") as f:
    data = json.load(f)

items = data["incorrect_examples"]["3"]

print("Incorrect 3-hop examples:", len(items))

for i, item in enumerate(items, 1):
    print(f"\n{i}. Q: {item['question']}")
    print("   Gold:", item["gold_answers"])
    print("   Predicted:", item["predicted_answers"])
    print("   Empty:", item["empty"])
