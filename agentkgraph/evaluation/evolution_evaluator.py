from dataclasses import dataclass,field


@dataclass
class EvolutionEvaluation:
    before:dict=field(default_factory=dict)
    after:dict=field(default_factory=dict)
    delta:dict=field(default_factory=dict)
    kg_changes:int=0

    def to_dict(self):
        return {
            "before":self.before,
            "after":self.after,
            "delta":self.delta,
            "kg_changes":self.kg_changes,
        }


class EvolutionEvaluator:
    def __init__(self):
        self.metric_names=[
            "exact_match",
            "token_f1",
            "hits_at_1",
            "precision_at_k",
            "provenance_completeness",
        ]

    def evaluate(self,before,after,kg_changes=0):
        before=dict(before or {})
        after=dict(after or {})

        delta={}

        for name in self.metric_names:
            b=float(before.get(name,0.0))
            a=float(after.get(name,0.0))
            delta[name]=round(a-b,10)

        return EvolutionEvaluation(
            before=before,
            after=after,
            delta=delta,
            kg_changes=int(kg_changes),
        ).to_dict()

    def improvement(self,before,after):
        result=self.evaluate(before,after)

        return {
            name:value
            for name,value in result["delta"].items()
        }

    def improved_metrics(self,before,after):
        delta=self.improvement(before,after)

        return [
            name
            for name,value in delta.items()
            if value>0
        ]

    def degraded_metrics(self,before,after):
        delta=self.improvement(before,after)

        return [
            name
            for name,value in delta.items()
            if value<0
        ]

    def summary(self,before,after,kg_changes=0):
        result=self.evaluate(
            before,
            after,
            kg_changes,
        )

        improved=self.improved_metrics(
            before,
            after,
        )

        degraded=self.degraded_metrics(
            before,
            after,
        )

        result["summary"]={
            "improved_metrics":improved,
            "degraded_metrics":degraded,
            "num_improved":len(improved),
            "num_degraded":len(degraded),
            "kg_changes":int(kg_changes),
        }

        return result