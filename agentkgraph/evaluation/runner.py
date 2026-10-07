import json
from dataclasses import dataclass,field
from pathlib import Path

from agentkgraph.evaluation.metrics import (
    exact_match,
    token_f1,
    hits_at_1,
    precision_at_k,
    provenance_completeness,
    routing_accuracy,
)


@dataclass
class EvaluationResult:
    count:int=0
    exact_match:float=0.0
    token_f1:float=0.0
    hits_at_1:float=0.0
    precision_at_k:float=0.0
    provenance_completeness:float=0.0
    routing_accuracy:float=0.0
    details:list=field(default_factory=list)

    def to_dict(self):
        return {
            "count":self.count,
            "exact_match":self.exact_match,
            "token_f1":self.token_f1,
            "hits_at_1":self.hits_at_1,
            "precision_at_k":self.precision_at_k,
            "provenance_completeness":self.provenance_completeness,
            "routing_accuracy":self.routing_accuracy,
            "details":self.details,
        }


class EvaluationRunner:
    def __init__(self,k=5):
        if k<=0:
            raise ValueError("k must be positive.")
        self.k=k

    def evaluate_example(
        self,
        prediction,
        references,
        retrieved=None,
        provenance=None,
        required_provenance=0,
        predicted_route=None,
        expected_route=None,
    ):
        retrieved=retrieved or []

        result={
            "exact_match":exact_match(
                prediction,
                references,
            ),
            "token_f1":token_f1(
                prediction,
                references,
            ),
            "hits_at_1":hits_at_1(
                retrieved,
                references,
            ),
            "precision_at_k":precision_at_k(
                retrieved,
                references,
                self.k,
            ),
            "provenance_completeness":provenance_completeness(
                provenance,
                required_provenance,
            ),
        }

        if predicted_route is not None and expected_route is not None:
            result["routing_accuracy"]=routing_accuracy(
                [predicted_route],
                [expected_route],
            )
        else:
            result["routing_accuracy"]=0.0

        return result

    def evaluate(self,examples):
        examples=list(examples)

        if not examples:
            return EvaluationResult().to_dict()

        totals={
            "exact_match":0.0,
            "token_f1":0.0,
            "hits_at_1":0.0,
            "precision_at_k":0.0,
            "provenance_completeness":0.0,
            "routing_accuracy":0.0,
        }

        details=[]

        for example in examples:
            result=self.evaluate_example(**example)

            for key in totals:
                totals[key]+=result[key]

            details.append(result)

        count=len(examples)

        return EvaluationResult(
            count=count,
            exact_match=totals["exact_match"]/count,
            token_f1=totals["token_f1"]/count,
            hits_at_1=totals["hits_at_1"]/count,
            precision_at_k=totals["precision_at_k"]/count,
            provenance_completeness=totals["provenance_completeness"]/count,
            routing_accuracy=totals["routing_accuracy"]/count,
            details=details,
        ).to_dict()

    def evaluate_jsonl(self,path,output_path=None):
        path=Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        examples=[]

        with path.open("r",encoding="utf-8") as f:
            for line_number,line in enumerate(f,1):
                line=line.strip()

                if not line:
                    continue

                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}: {exc}"
                    ) from exc

        result=self.evaluate(examples)

        if output_path is not None:
            output_path=Path(output_path)
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with output_path.open("w",encoding="utf-8") as f:
                json.dump(
                    result,
                    f,
                    indent=2,
                )

        return result