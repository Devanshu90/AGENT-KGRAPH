from dataclasses import dataclass,field

from agentkgraph.data.dataset_integration import DatasetIntegration
from agentkgraph.evaluation.metrics import (
    exact_match,
    token_f1,
)


@dataclass
class DatasetEvaluation:
    dataset:str
    split:str
    count:int=0
    exact_match:float=0.0
    token_f1:float=0.0
    hop_metrics:dict=field(default_factory=dict)


class DatasetEvaluator:
    def __init__(self,loader=None):
        self.loader=loader or DatasetIntegration()

    def evaluate_records(self,records,predictions):
        records=list(records)
        predictions=list(predictions)

        if len(records)!=len(predictions):
            raise ValueError(
                "Number of predictions must match number of records."
            )

        if not records:
            return {
                "count":0,
                "exact_match":0.0,
                "token_f1":0.0,
            }

        em_total=0.0
        f1_total=0.0

        for record,prediction in zip(records,predictions):
            em_total+=exact_match(
                prediction,
                record.answers,
            )
            f1_total+=token_f1(
                prediction,
                record.answers,
            )

        count=len(records)

        result={
            "count":count,
            "exact_match":em_total/count,
            "token_f1":f1_total/count,
        }

        return result

    def evaluate_dataset(
        self,
        dataset,
        split,
        predictions,
        limit=None,
    ):
        records=self.loader.load(
            dataset,
            split,
        )

        if limit is not None:
            if limit<=0:
                raise ValueError("limit must be positive.")
            records=records[:limit]

        if len(predictions)!=len(records):
            raise ValueError(
                f"Expected {len(records)} predictions, "
                f"received {len(predictions)}."
            )

        result=self.evaluate_records(
            records,
            predictions,
        )

        result["dataset"]=dataset
        result["split"]=split

        if dataset.lower()=="metaqa":
            result["hop_metrics"]=self._evaluate_hops(
                records,
                predictions,
            )

        return result

    def _evaluate_hops(self,records,predictions):
        groups={}

        for record,prediction in zip(records,predictions):
            hop=record.hop

            if hop not in groups:
                groups[hop]={
                    "count":0,
                    "exact_match":0.0,
                    "token_f1":0.0,
                }

            groups[hop]["count"]+=1
            groups[hop]["exact_match"]+=exact_match(
                prediction,
                record.answers,
            )
            groups[hop]["token_f1"]+=token_f1(
                prediction,
                record.answers,
            )

        for hop,result in groups.items():
            count=result["count"]

            if count:
                result["exact_match"]/=count
                result["token_f1"]/=count

        return groups

    def evaluate_fixed_prediction(
        self,
        dataset,
        split,
        prediction,
        limit=10,
    ):
        records=self.loader.load(
            dataset,
            split,
        )

        records=records[:limit]

        predictions=[prediction for _ in records]

        return self.evaluate_dataset(
            dataset,
            split,
            predictions,
            limit=limit,
        )