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
            self.model=CrossEncoder(
                self.model_name
            )

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

        predicate=predicate.replace(
            "_",
            " "
        )

        return predicate

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

        if predicate in {
            "was released in",
            "has genre",
            "has tag"
        }:
            return (
                f"{subject} "
                f"{predicate} "
                f"{obj}."
            )

        if predicate in {
            "directed",
            "starred",
            "produced by",
            "is married to"
        }:
            return (
                f"{subject} "
                f"{predicate} "
                f"{obj}."
            )

        if predicate in {
            "was written by",
            "was produced by"
        }:
            return (
                f"{subject} "
                f"{predicate} "
                f"{obj}."
            )

        return (
            f"{subject} "
            f"{predicate} "
            f"{obj}."
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

        scores=np.asarray(
            scores[0],
            dtype=np.float32
        )

        if scores.ndim==0:
            confidence=float(
                1.0/
                (
                    1.0+
                    np.exp(
                        -float(scores)
                    )
                )
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

            return VerificationResult(
                accepted=accepted,
                confidence=confidence,
                reason=(
                    "The passage entails the "
                    "extracted triple."
                    if accepted
                    else
                    "The passage does not sufficiently "
                    "entail the extracted triple."
                )
            )

        labels=[
            "contradiction",
            "entailment",
            "neutral"
        ]

        if hasattr(
            self.model.model.config,
            "id2label"
        ):
            mapping=self.model.model.config.id2label

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

        entailment_index=None

        for i,label in enumerate(labels):
            if "entail" in label:
                entailment_index=i
                break

        if entailment_index is None:
            entailment_index=1

        shifted=scores-np.max(
            scores
        )

        probabilities=(
            np.exp(shifted)/
            np.sum(
                np.exp(shifted)
            )
        )

        confidence=float(
            probabilities[
                entailment_index
            ]
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

        if accepted:
            reason=(
                "The passage entails the "
                "extracted triple."
            )
        else:
            reason=(
                "The passage does not sufficiently "
                "entail the extracted triple."
            )

        return VerificationResult(
            accepted=accepted,
            confidence=confidence,
            reason=reason
        )

    def _dry_verify(self,triple,text):
        s=triple.subject.lower()
        o=triple.object.lower()
        t=text.lower()

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