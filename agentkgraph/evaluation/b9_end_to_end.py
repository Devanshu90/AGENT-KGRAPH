
import gc
import json
import re
import statistics
import time
from pathlib import Path

from agentkgraph.config import Config
from agentkgraph.evaluation.metrics import exact_match, token_f1
from agentkgraph.pipeline import Engine


def metaqa_set_exact_match(prediction, answers):
    def normalize(value):
        return re.sub(r"\s+", " ", str(value or "")).strip().casefold()

    if isinstance(answers, str):
        answers = [answers]

    predicted = {
        normalize(part)
        for part in str(prediction or "").split(",")
        if normalize(part)
    }
    expected = {
        normalize(answer)
        for answer in answers
        if normalize(answer)
    }

    return float(predicted == expected)


class B9EndToEndBenchmark:
    def __init__(self, split="test", samples_per_hop=2, output_dir="reports/b9"):
        self.split = split
        self.samples_per_hop = samples_per_hop
        self.output_dir = Path(output_dir)

    def _load_records(self):
        path = Path(f"data/processed/metaqa/{self.split}.jsonl")
        groups = {1: [], 2: [], 3: []}

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                hop = int(record["hop"])

                if hop in groups and len(groups[hop]) < self.samples_per_hop:
                    groups[hop].append(record)

                if all(len(groups[x]) >= self.samples_per_hop for x in (1, 2, 3)):
                    break

        records = []

        for hop in (1, 2, 3):
            records.extend(groups[hop])

        return records

    def _build_engine(self):
        config = Config()
        config.kg.hop_limit = 3
        config.models.synthesizer_max_new_tokens = 64
        config.models.synthesizer_temperature = 0.0

        print("Building Engine without full corpus...")
        engine = Engine(
            passages=[],
            config=config,
            dry_run=False
        )

        print("Loading MetaQA KG...")
        engine.load_kg("data/kg/metaqa_graph.json")

        return engine

    def _evaluate(self, prediction, answers):
        return {
            "exact_match": metaqa_set_exact_match(prediction, answers),
            "token_f1": token_f1(prediction, answers)
        }

    def run(self):
        records = self._load_records()

        print()
        print(f"Selected {len(records)} MetaQA questions:")

        for hop in (1, 2, 3):
            count = sum(1 for r in records if int(r["hop"]) == hop)
            print(f"  {hop}-hop: {count}")

        engine = self._build_engine()
        results = []

        try:
            total_start = time.perf_counter()

            for index, record in enumerate(records, 1):
                question = record["question"]
                answers = record["answers"]
                hop = int(record["hop"])

                print()
                print(f"[{index}/{len(records)}] {hop}-hop: {question}")

                start = time.perf_counter()
                answer_result = engine.answer(question, evolve=False)

                route = answer_result.get("route")
                evidence = answer_result.get("evidence", [])
                answer = answer_result.get("answer", "")
                elapsed = (time.perf_counter() - start) * 1000

                metrics = self._evaluate(answer, answers)

                item = {
                    "question_id": record["question_id"],
                    "hop": hop,
                    "question": question,
                    "answers": answers,
                    "prediction": answer,
                    "route": route,
                    "evidence_count": len(evidence),
                    "exact_match": metrics["exact_match"],
                    "token_f1": metrics["token_f1"],
                    "latency_ms": elapsed
                }

                results.append(item)

                print(f"  route: {route}")
                print(f"  evidence: {len(evidence)}")
                print(f"  EM: {metrics['exact_match']:.3f}")
                print(f"  F1: {metrics['token_f1']:.3f}")
                print(f"  latency: {elapsed:.2f} ms")
                print(f"  answer: {answer!r}")

            total_elapsed = (time.perf_counter() - total_start) * 1000

        finally:
            engine.unload_all()
            gc.collect()

        overall = {
            "exact_match": statistics.mean(
                r["exact_match"] for r in results
            ) if results else 0.0,
            "token_f1": statistics.mean(
                r["token_f1"] for r in results
            ) if results else 0.0,
            "average_latency_ms": statistics.mean(
                r["latency_ms"] for r in results
            ) if results else 0.0,
            "total_latency_ms": total_elapsed if results else 0.0
        }

        hop_metrics = {}

        for hop in (1, 2, 3):
            subset = [r for r in results if r["hop"] == hop]

            if subset:
                hop_metrics[str(hop)] = {
                    "count": len(subset),
                    "exact_match": statistics.mean(
                        r["exact_match"] for r in subset
                    ),
                    "token_f1": statistics.mean(
                        r["token_f1"] for r in subset
                    ),
                    "average_latency_ms": statistics.mean(
                        r["latency_ms"] for r in subset
                    )
                }

        route_distribution = {}

        for result in results:
            route = result["route"] or "unknown"
            route_distribution[route] = route_distribution.get(route, 0) + 1

        report = {
            "benchmark": "B9 real-model end-to-end MetaQA",
            "split": self.split,
            "samples_per_hop": self.samples_per_hop,
            "sample_count": len(results),
            "settings": {
                "synthesizer_max_new_tokens": 64,
                "synthesizer_temperature": 0.0,
                "vector_corpus_loaded": False,
                "kg_path": "data/kg/metaqa_graph.json",
                "exact_match_method": "normalized answer-set equality"
            },
            "overall": overall,
            "hop_metrics": hop_metrics,
            "route_distribution": route_distribution,
            "results": results
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.output_dir / "b9_end_to_end_metaqa.json"
        txt_path = self.output_dir / "b9_end_to_end_metaqa.txt"

        json_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        lines = [
            "B9 REAL-MODEL END-TO-END METAQA",
            "=" * 60,
            f"Split: {self.split}",
            f"Samples: {len(results)}",
            "",
            "Overall:",
            f"  Exact Match: {overall['exact_match']:.4f}",
            f"  Token F1: {overall['token_f1']:.4f}",
            f"  Average latency: {overall['average_latency_ms']:.2f} ms",
            "",
            "Hop Metrics:"
        ]

        for hop, metrics in hop_metrics.items():
            lines.extend([
                f"  {hop}-hop:",
                f"    Count: {metrics['count']}",
                f"    EM: {metrics['exact_match']:.4f}",
                f"    F1: {metrics['token_f1']:.4f}",
                f"    Latency: {metrics['average_latency_ms']:.2f} ms"
            ])

        lines.extend([
            "",
            "Route Distribution:",
            json.dumps(route_distribution, indent=2),
            "",
            "Per-question Results:"
        ])

        for result in results:
            lines.extend([
                "",
                f"Question ID: {result['question_id']}",
                f"Hop: {result['hop']}",
                f"Route: {result['route']}",
                f"Evidence: {result['evidence_count']}",
                f"EM: {result['exact_match']:.4f}",
                f"F1: {result['token_f1']:.4f}",
                f"Latency: {result['latency_ms']:.2f} ms",
                f"Prediction: {result['prediction']}"
            ])

        txt_path.write_text("\n".join(lines), encoding="utf-8")

        print()
        print("=" * 60)
        print("B9 END-TO-END BENCHMARK COMPLETE")
        print("=" * 60)
        print(f"Exact Match: {overall['exact_match']:.4f}")
        print(f"Token F1: {overall['token_f1']:.4f}")
        print(f"Average latency: {overall['average_latency_ms']:.2f} ms")
        print(f"Routes: {route_distribution}")
        print(f"JSON: {json_path}")
        print(f"TXT: {txt_path}")

        return report


def main():
    benchmark = B9EndToEndBenchmark(
        split="test",
        samples_per_hop=2
    )
    benchmark.run()


if __name__ == "__main__":
    main()
