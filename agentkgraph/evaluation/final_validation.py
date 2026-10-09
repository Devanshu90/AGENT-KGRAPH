import gc
import json
import platform
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path

from agentkgraph.config import Config
from agentkgraph.evaluation.ablation import AblationEvaluator
from agentkgraph.evaluation.dataset_evaluator import DatasetEvaluator
from agentkgraph.evaluation.evolution_evaluator import EvolutionEvaluator
from agentkgraph.evaluation.retrieval_evaluator import RetrievalEvaluator
from agentkgraph.evaluation.runner import EvaluationRunner


@dataclass
class ValidationCheck:
    name: str
    status: str
    duration_ms: float = 0.0
    details: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self):
        return {
            "name": self.name,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 3),
            "details": self.details,
            "error": self.error,
        }


class FinalValidator:
    def __init__(
        self,
        config=None,
        dry_run=True,
        output_dir="reports/b9",
    ):
        self.config = config or Config()
        self.dry_run = dry_run
        self.output_dir = Path(output_dir)
        self.checks = []

    def _run_check(self, name, fn):
        started = time.perf_counter()

        try:
            result = fn()

            duration = (
                time.perf_counter() - started
            ) * 1000

            check = ValidationCheck(
                name=name,
                status="PASS",
                duration_ms=duration,
                details=result or {},
            )

        except Exception as exc:
            duration = (
                time.perf_counter() - started
            ) * 1000

            check = ValidationCheck(
                name=name,
                status="FAIL",
                duration_ms=duration,
                error=f"{type(exc).__name__}: {exc}",
            )

        self.checks.append(check)
        return check

    def validate_environment(self):
        def check():
            import networkx
            import torch

            return {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "networkx": networkx.__version__,
                "torch": torch.__version__,
                "cuda_available": bool(
                    torch.cuda.is_available()
                ),
                "dry_run": self.dry_run,
                "project_root": str(Path.cwd()),
            }

        return self._run_check(
            "environment",
            check,
        )

    def validate_configuration(self):
        def check():
            models = self.config.models

            return {
                "extractor": models.extractor,
                "verifier": models.verifier,
                "router": models.router,
                "synthesizer": models.synthesizer,
                "embedder": models.embedder,
                "routing_actions": list(
                    self.config.routing.actions
                ),
                "ucb_beta": self.config.routing.ucb_beta,
                "min_explore": self.config.routing.min_explore,
            }

        return self._run_check(
            "configuration",
            check,
        )

    def validate_metrics(self):
        def check():
            runner = EvaluationRunner(k=3)

            result = runner.evaluate(
                [
                    {
                        "prediction": "Alice",
                        "references": ["Alice"],
                        "retrieved": ["Alice", "Bob"],
                        "provenance": ["p1"],
                        "required_provenance": 1,
                        "predicted_route": "vector",
                        "expected_route": "vector",
                    }
                ]
            )

            if result["exact_match"] != 1.0:
                raise AssertionError(
                    "Exact Match metric validation failed."
                )

            if result["token_f1"] != 1.0:
                raise AssertionError(
                    "Token-F1 metric validation failed."
                )

            if result["hits_at_1"] != 1.0:
                raise AssertionError(
                    "Hits@1 metric validation failed."
                )

            if result["precision_at_k"] != 0.5:
                raise AssertionError(
                    "Precision@k metric validation failed."
                )

            if result["provenance_completeness"] != 1.0:
                raise AssertionError(
                    "Provenance metric validation failed."
                )

            if result["routing_accuracy"] != 1.0:
                raise AssertionError(
                    "Routing metric validation failed."
                )

            return result

        return self._run_check(
            "evaluation_metrics",
            check,
        )

    def validate_dataset_evaluator(self):
        def check():
            evaluator = DatasetEvaluator()

            records = [
                type(
                    "Record",
                    (),
                    {
                        "answers": ["Paris"],
                        "hop": 1,
                    },
                )(),
                type(
                    "Record",
                    (),
                    {
                        "answers": ["London"],
                        "hop": 2,
                    },
                ),
            ]

            result = evaluator.evaluate_records(
                records,
                ["Paris", "London"],
            )

            if result["exact_match"] != 1.0:
                raise AssertionError(
                    "Dataset evaluator EM failed."
                )

            if result["token_f1"] != 1.0:
                raise AssertionError(
                    "Dataset evaluator Token-F1 failed."
                )

            return result

        return self._run_check(
            "dataset_evaluator",
            check,
        )

    def validate_retrieval_evaluator(self):
        def check():
            evaluator = RetrievalEvaluator(k=3)

            result = evaluator.evaluate(
                [
                    {
                        "retrieved": [
                            "Alice",
                            "Bob",
                            "Carol",
                        ],
                        "references": ["Alice"],
                        "predicted_route": "hybrid",
                        "expected_route": "hybrid",
                    }
                ]
            )

            if result["hits_at_1"] != 1.0:
                raise AssertionError(
                    "Retrieval Hits@1 failed."
                )

            if result["precision_at_k"] <= 0.0:
                raise AssertionError(
                    "Retrieval Precision@k failed."
                )

            if result["routing_accuracy"] != 1.0:
                raise AssertionError(
                    "Retrieval routing accuracy failed."
                )

            return result

        return self._run_check(
            "retrieval_evaluator",
            check,
        )

    def validate_ablation(self):
        def check():
            evaluator = AblationEvaluator(
                baseline="full"
            )

            configurations = {
                "full": {
                    "exact_match": 0.80,
                    "token_f1": 0.85,
                },
                "no_router": {
                    "exact_match": 0.70,
                    "token_f1": 0.78,
                },
                "vector_only": {
                    "exact_match": 0.65,
                    "token_f1": 0.74,
                },
            }

            result = evaluator.evaluate(
                configurations
            )

            if (
                result["deltas"]["no_router"]["exact_match"]
                >= 0
            ):
                raise AssertionError(
                    "Ablation delta validation failed."
                )

            best = evaluator.best_configuration(
                configurations,
                "exact_match",
            )

            if best != "full":
                raise AssertionError(
                    "Ablation ranking validation failed."
                )

            return result

        return self._run_check(
            "ablation",
            check,
        )

    def validate_evolution(self):
        def check():
            evaluator = EvolutionEvaluator()

            before = {
                "exact_match": 0.60,
                "token_f1": 0.65,
                "hits_at_1": 0.70,
                "precision_at_k": 0.55,
                "provenance_completeness": 0.50,
            }

            after = {
                "exact_match": 0.70,
                "token_f1": 0.72,
                "hits_at_1": 0.75,
                "precision_at_k": 0.62,
                "provenance_completeness": 0.80,
            }

            result = evaluator.summary(
                before,
                after,
                kg_changes=3,
            )

            if result["delta"]["exact_match"] <= 0:
                raise AssertionError(
                    "Evolution improvement validation failed."
                )

            if result["kg_changes"] != 3:
                raise AssertionError(
                    "KG change tracking failed."
                )

            return result

        return self._run_check(
            "evolution_evaluator",
            check,
        )

    def validate_engine(self):
        def check():
            from agentkgraph.kg.graph_store import (
                GraphStore,
                Triple,
            )
            from agentkgraph.pipeline import Engine

            import tempfile

            kg = GraphStore(self.config)
            now = time.time()

            triples = [
                Triple(
                    "Alice",
                    "works_at",
                    "OpenAI",
                    "p1",
                    0.95,
                    now,
                ),
                Triple(
                    "OpenAI",
                    "located_in",
                    "USA",
                    "p2",
                    0.95,
                    now,
                ),
                Triple(
                    "Bob",
                    "works_at",
                    "Google",
                    "p3",
                    0.95,
                    now,
                ),
            ]

            committed = 0

            for triple in triples:
                if kg.commit(triple):
                    committed += 1

            if committed != 3:
                raise AssertionError(
                    "Temporary KG construction failed."
                )

            with tempfile.TemporaryDirectory() as temp_dir:
                kg_path = Path(temp_dir) / "kg.json"
                kg.save(kg_path)

                passages = [
                    {
                        "passage_id": "p1",
                        "text": "Alice works at OpenAI.",
                    },
                    {
                        "passage_id": "p2",
                        "text": "OpenAI is located in USA.",
                    },
                    {
                        "passage_id": "p3",
                        "text": "Bob works at Google.",
                    },
                ]

                engine = Engine(
                    passages=passages,
                    config=self.config,
                    dry_run=self.dry_run,
                )

                engine.load_kg(
                    str(kg_path)
                )

                retrieval = engine.retrieve(
                    "Where does Alice work?"
                )

                if retrieval["route"] not in {
                    "vector",
                    "kg",
                    "hybrid",
                }:
                    raise AssertionError(
                        "Invalid router action."
                    )

                if not isinstance(
                    retrieval["evidence"],
                    list,
                ):
                    raise AssertionError(
                        "Retrieval evidence is not a list."
                    )

                answer = engine.answer(
                    "Where does Alice work?",
                    evolve=False,
                )

                if "answer" not in answer:
                    raise AssertionError(
                        "Engine answer is missing."
                    )

                if "route" not in answer:
                    raise AssertionError(
                        "Engine answer route is missing."
                    )

                engine.unload_all()

            return {
                "kg_triples": committed,
                "route": retrieval["route"],
                "evidence_count": len(
                    retrieval["evidence"]
                ),
                "answer_present": bool(
                    answer["answer"]
                ),
            }

        return self._run_check(
            "engine_end_to_end",
            check,
        )

    def validate_router_learning(self):
        def check():
            from agentkgraph.routing.router import Router

            router = Router(
                config=self.config.routing,
                model_name=self.config.models.router,
                dry_run=True,
            )

            query = "test routing query"

            valid = set(
                self.config.routing.actions
            )

            expected = {
                "vector",
                "kg",
                "hybrid",
            }

            if valid != expected:
                raise AssertionError(
                    "Expected vector, kg and hybrid "
                    "routing actions."
                )

            actions = []

            for _ in range(
                self.config.routing.min_explore
            ):
                action = router.route(query)
                actions.append(action)

                if action not in valid:
                    raise AssertionError(
                        f"Invalid router action: {action}"
                    )

                reward = (
                    1.0
                    if action == "kg"
                    else 0.0
                )

                router.update(
                    query,
                    action,
                    reward,
                )

            if not actions:
                raise AssertionError(
                    "Router produced no actions."
                )

            state = router.update(
                query,
                "kg",
                1.0,
            )

            return {
                "configured_actions": sorted(valid),
                "actions_seen": sorted(
                    set(actions)
                ),
                "updates": self.config.routing.min_explore,
                "all_actions_configured": True,
                "final_update": state,
            }

        return self._run_check(
            "ucb_router",
            check,
        )

    def validate_resources(self):
        def check():
            from agentkgraph.kg.graph_store import (
                GraphStore,
                Triple,
            )
            from agentkgraph.pipeline import Engine

            import tempfile

            kg = GraphStore(self.config)
            now = time.time()

            committed = kg.commit(
                Triple(
                    "Alice",
                    "works_at",
                    "OpenAI",
                    "p1",
                    0.95,
                    now,
                )
            )

            if not committed:
                raise AssertionError(
                    "Resource measurement KG setup failed."
                )

            with tempfile.TemporaryDirectory() as temp_dir:
                kg_path = Path(temp_dir) / "kg.json"
                kg.save(kg_path)

                passages = [
                    {
                        "passage_id": "p1",
                        "text": "Alice works at OpenAI.",
                    }
                ]

                engine = Engine(
                    passages=passages,
                    config=self.config,
                    dry_run=True,
                )

                engine.load_kg(
                    str(kg_path)
                )

                gc.collect()
                tracemalloc.start()

                started = time.perf_counter()

                retrieval = engine.retrieve(
                    "Where does Alice work?"
                )

                retrieve_ms = (
                    time.perf_counter() - started
                ) * 1000

                current, peak = (
                    tracemalloc.get_traced_memory()
                )

                tracemalloc.stop()

                engine.unload_all()

                if not retrieval["route"]:
                    raise AssertionError(
                        "Engine retrieval returned no route."
                    )

                return {
                    "measurement": "engine_retrieval",
                    "duration_ms": round(
                        retrieve_ms,
                        3,
                    ),
                    "peak_memory_kb": round(
                        peak / 1024,
                        3,
                    ),
                    "route": retrieval["route"],
                    "evidence_count": len(
                        retrieval["evidence"]
                    ),
                }

        return self._run_check(
            "resource_measurement",
            check,
        )

    def run(self):
        self.checks = []

        self.validate_environment()
        self.validate_configuration()
        self.validate_metrics()
        self.validate_dataset_evaluator()
        self.validate_retrieval_evaluator()
        self.validate_ablation()
        self.validate_evolution()
        self.validate_router_learning()
        self.validate_engine()
        self.validate_resources()

        passed = sum(
            check.status == "PASS"
            for check in self.checks
        )

        failed = sum(
            check.status == "FAIL"
            for check in self.checks
        )

        result = {
            "validation_mode": (
                "dry_run"
                if self.dry_run
                else "real_model"
            ),
            "real_model_benchmark_status": (
                "not_run"
                if self.dry_run
                else "requested"
            ),
            "passed": passed,
            "failed": failed,
            "total": len(self.checks),
            "success": failed == 0,
            "checks": [
                check.to_dict()
                for check in self.checks
            ],
        }

        return result

    def save_reports(self, result):
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path = (
            self.output_dir
            / "final_validation.json"
        )

        text_path = (
            self.output_dir
            / "final_validation.txt"
        )

        with json_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                result,
                f,
                indent=2,
                ensure_ascii=False,
            )

        lines = [
            "AGENT-KGRAPH B9 FINAL VALIDATION",
            "=" * 36,
            "",
            (
                "Validation mode: "
                f"{result['validation_mode']}"
            ),
            (
                "Real model benchmark: "
                f"{result['real_model_benchmark_status']}"
            ),
            f"Passed: {result['passed']}",
            f"Failed: {result['failed']}",
            f"Total: {result['total']}",
            f"Overall success: {result['success']}",
            "",
        ]

        for check in result["checks"]:
            lines.append(
                f"[{check['status']}] "
                f"{check['name']} "
                f"({check['duration_ms']} ms)"
            )

            if check["error"]:
                lines.append(
                    f"  Error: {check['error']}"
                )

            if check["details"]:
                lines.append(
                    "  Details: "
                    + json.dumps(
                        check["details"],
                        ensure_ascii=False,
                    )
                )

        with text_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                "\n".join(lines)
            )

        return {
            "json": str(json_path),
            "text": str(text_path),
        }


def run_final_validation(
    dry_run=True,
    output_dir="reports/b9",
):
    validator = FinalValidator(
        dry_run=dry_run,
        output_dir=output_dir,
    )

    result = validator.run()

    reports = validator.save_reports(
        result
    )

    result["reports"] = reports

    return result


if __name__ == "__main__":
    result = run_final_validation(
        dry_run=True
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )