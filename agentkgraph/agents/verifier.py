from dataclasses import dataclass

import numpy as np
from sentence_transformers import CrossEncoder


@dataclass
class VerificationResult:
    accepted:bool
    confidence:float
    reason:str


class Verifier:
    def __init__(
        self,
        threshold=0.60,
        model_name="cross-encoder/nli-deberta-v3-base",
        dry_run=False
    ):
        self.threshold=threshold
        self.model_name=model_name
        self.dry_run=dry_run
        self.model=None

        if not self.dry_run:
            self._load_model()

    def _load_model(self):
        self.model=CrossEncoder(
            self.model_name
        )

        self.model.model.eval()

    def _predicate_to_text(self,predicate):
        predicate=predicate.strip()

        mapping={
            "directed_by":"directed",
            "written_by":"was written by",
            "starred_actors":"starred",
            "has_genre":"has genre",
            "release_year":"was released in",
            "born_in":"was born in",
            "married_to":"is married to",
            "has_tags":"has tag",
            "produced_by":"was produced by",
            "music_by":"has music by"
        }

        if predicate in mapping:
            return mapping[predicate]

        return predicate.replace(
            "_",
            " "
        )

    def _build_hypothesis(self,triple):
        subject=str(
            triple.subject
        ).strip()

        predicate=self._predicate_to_text(
            str(triple.predicate)
        )

        obj=str(
            triple.object
        ).strip()

        return (
            f"{subject} "
            f"{predicate} "
            f"{obj}."
        )

    def _get_labels(self,scores):
        labels=[
            "contradiction",
            "entailment",
            "neutral"
        ]

        try:
            mapping=self.model.model.config.id2label

            if mapping:
                labels=[
                    str(
                        mapping.get(
                            i,
                            str(i)
                        )
                    ).lower()
                    for i in range(
                        len(scores)
                    )
                ]
        except (
            AttributeError,
            TypeError
        ):
            pass

        return labels

    def _entailment_confidence(self,scores):
        scores=np.asarray(
            scores,
            dtype=np.float32
        )

        if scores.ndim==0:
            value=float(
                scores
            )

            return float(
                1.0/
                (
                    1.0+
                    np.exp(-value)
                )
            )

        scores=scores.reshape(-1)

        if len(scores)==1:
            value=float(
                scores[0]
            )

            return float(
                1.0/
                (
                    1.0+
                    np.exp(-value)
                )
            )

        labels=self._get_labels(
            scores
        )

        entailment_index=None

        for i,label in enumerate(labels):
            if "entail" in label:
                entailment_index=i
                break

        if entailment_index is None:
            entailment_index=1 if len(scores)>1 else 0

        shifted=scores-np.max(
            scores
        )

        probabilities=(
            np.exp(shifted)/
            np.sum(
                np.exp(shifted)
            )
        )

        return float(
            probabilities[
                entailment_index
            ]
        )

    def verify(self,triple,text):
        if self.dry_run:
            return self._dry_verify(
                triple,
                text
            )

        hypothesis=self._build_hypothesis(
            triple
        )

        scores=self.model.predict(
            [(text,hypothesis)]
        )

        confidence=self._entailment_confidence(
            scores[0]
        )

        confidence=max(
            0.0,
            min(
                1.0,
                confidence
            )
        )

        accepted=(
            confidence>=self.threshold
        )

        reason=(
            "The passage entails the "
            "extracted triple."
            if accepted
            else
            "The passage does not sufficiently "
            "entail the extracted triple."
        )

        return VerificationResult(
            accepted=accepted,
            confidence=confidence,
            reason=reason
        )

    def _dry_verify(self,triple,text):
        s=str(
            triple.subject
        ).lower()

        o=str(
            triple.object
        ).lower()

        t=str(
            text
        ).lower()

        supported=(
            s in t and
            o in t
        )

        if supported:
            return VerificationResult(
                accepted=True,
                confidence=0.90,
                reason=(
                    "Subject and object are supported "
                    "by the passage."
                )
            )

        return VerificationResult(
            accepted=False,
            confidence=0.10,
            reason=(
                "Triple is not sufficiently supported "
                "by the passage."
            )
        )

    def unload(self):
        if self.model is not None:
            del self.model
            self.model=None
