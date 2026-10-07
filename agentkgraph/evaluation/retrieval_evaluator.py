from dataclasses import dataclass,field

from agentkgraph.evaluation.metrics import (
    hits_at_1,
    precision_at_k,
    routing_accuracy,
)


@dataclass
class RetrievalEvaluation:
    count:int=0
    hits_at_1:float=0.0
    precision_at_k:float=0.0
    routing_accuracy:float=0.0
    by_method:dict=field(default_factory=dict)

    def to_dict(self):
        return {
            "count":self.count,
            "hits_at_1":self.hits_at_1,
            "precision_at_k":self.precision_at_k,
            "routing_accuracy":self.routing_accuracy,
            "by_method":self.by_method,
        }


class RetrievalEvaluator:
    def __init__(self,k=5):
        if k<=0:
            raise ValueError("k must be positive.")
        self.k=k

    def evaluate_example(
        self,
        retrieved,
        references,
        predicted_route=None,
        expected_route=None,
    ):
        result={
            "hits_at_1":hits_at_1(
                retrieved,
                references,
            ),
            "precision_at_k":precision_at_k(
                retrieved,
                references,
                self.k,
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
            return RetrievalEvaluation().to_dict()

        totals={
            "hits_at_1":0.0,
            "precision_at_k":0.0,
            "routing_accuracy":0.0,
        }

        for example in examples:
            result=self.evaluate_example(**example)

            for key in totals:
                totals[key]+=result[key]

        count=len(examples)

        return RetrievalEvaluation(
            count=count,
            hits_at_1=totals["hits_at_1"]/count,
            precision_at_k=totals["precision_at_k"]/count,
            routing_accuracy=totals["routing_accuracy"]/count,
        ).to_dict()

    def evaluate_methods(self,examples_by_method):
        result={}

        for method,examples in examples_by_method.items():
            result[method]=self.evaluate(examples)

        return result

    def compare_methods(self,examples_by_method):
        evaluations=self.evaluate_methods(
            examples_by_method
        )

        comparison={}

        for method,result in evaluations.items():
            comparison[method]={
                "count":result["count"],
                "hits_at_1":result["hits_at_1"],
                "precision_at_k":result["precision_at_k"],
            }

        return comparison