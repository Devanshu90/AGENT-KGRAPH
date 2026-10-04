from dataclasses import dataclass


@dataclass
class VerificationResult:
    accepted:bool
    confidence:float
    reason:str


class Verifier:
    def __init__(self,threshold=0.60,dry_run=True):
        self.threshold=threshold
        self.dry_run=dry_run

    def verify(self,triple,text):
        if self.dry_run:
            return self._dry_verify(triple,text)

        raise NotImplementedError("Real NLI verifier will be added later.")

    def _dry_verify(self,triple,text):
        s=triple.subject.lower()
        p=triple.predicate.lower()
        o=triple.object.lower()
        t=text.lower()

        supported=s in t and o in t

        if supported:
            return VerificationResult(
                accepted=True,
                confidence=0.90,
                reason="Subject and object are supported by the passage."
            )

        return VerificationResult(
            accepted=False,
            confidence=0.10,
            reason="Triple is not sufficiently supported by the passage."
        )