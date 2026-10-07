from dataclasses import dataclass,field


@dataclass
class AblationResult:
    baseline:str
    configurations:dict=field(default_factory=dict)
    deltas:dict=field(default_factory=dict)

    def to_dict(self):
        return {
            "baseline":self.baseline,
            "configurations":self.configurations,
            "deltas":self.deltas,
        }


class AblationEvaluator:
    def __init__(self,baseline="full"):
        self.baseline=baseline

    def evaluate(self,configurations):
        configurations={
            name:dict(metrics or {})
            for name,metrics in configurations.items()
        }

        if self.baseline not in configurations:
            raise ValueError(
                f"Baseline '{self.baseline}' not found."
            )

        baseline_metrics=configurations[self.baseline]
        deltas={}

        for name,metrics in configurations.items():
            if name==self.baseline:
                continue

            delta={}

            keys=set(baseline_metrics)|set(metrics)

            for key in keys:
                baseline_value=float(
                    baseline_metrics.get(key,0.0)
                )
                value=float(
                    metrics.get(key,0.0)
                )

                delta[key]=round(
                    value-baseline_value,
                    10,
                )

            deltas[name]=delta

        return AblationResult(
            baseline=self.baseline,
            configurations=configurations,
            deltas=deltas,
        ).to_dict()

    def ranking(self,configurations,metric):
        rows=[]

        for name,metrics in configurations.items():
            rows.append(
                (
                    name,
                    float(metrics.get(metric,0.0)),
                )
            )

        rows.sort(
            key=lambda item:item[1],
            reverse=True,
        )

        return rows

    def best_configuration(self,configurations,metric):
        ranking=self.ranking(
            configurations,
            metric,
        )

        if not ranking:
            return None

        return ranking[0][0]

    def metric_impact(
        self,
        configurations,
        configuration,
        metric,
    ):
        result=self.evaluate(configurations)

        if configuration not in result["deltas"]:
            raise ValueError(
                f"Configuration '{configuration}' "
                "is the baseline or does not exist."
            )

        return result["deltas"][configuration].get(
            metric,
            0.0,
        )