import re
import time
from dataclasses import dataclass

from agentkgraph.kg.graph_store import Triple


@dataclass
class FeedbackResult:
    answer:str
    confidence:float
    consistent:bool
    evolved:bool
    triples_added:int
    citations:list
    reason:str


class FeedbackEngine:
    def __init__(
        self,
        synthesizer,
        extractor,
        verifier,
        merger,
        kg,
        consistency_n=2,
        threshold=0.80,
        user_feedback_weight=0.30,
        max_retry=2
    ):
        self.synthesizer=synthesizer
        self.extractor=extractor
        self.verifier=verifier
        self.merger=merger
        self.kg=kg
        self.consistency_n=consistency_n
        self.threshold=threshold
        self.user_feedback_weight=user_feedback_weight
        self.max_retry=max_retry

    def process(
        self,
        query,
        evidence,
        user_feedback=None
    ):
        answers=[]

        for _ in range(self.consistency_n):
            result=self.synthesizer.generate(
                query,
                evidence
            )

            if isinstance(result,dict):
                answer=result.get(
                    "answer",
                    ""
                )
            else:
                answer=str(result)

            answers.append(answer)

        confidence=self._consistency_score(
            answers
        )

        if user_feedback is not None:
            confidence=self._apply_user_feedback(
                confidence,
                user_feedback
            )

        answer=self._select_answer(
            answers
        )

        if confidence<self.threshold:
            return FeedbackResult(
                answer=answer,
                confidence=confidence,
                consistent=False,
                evolved=False,
                triples_added=0,
                citations=self._extract_citations(
                    answer
                ),
                reason=(
                    "Confidence below evolution "
                    "threshold. KG was not updated."
                )
            )

        triples=self._extract_candidates(
            answer,
            evidence
        )

        verified=[]

        evidence_text=self._evidence_text(
            evidence
        )

        verification_text=(
            answer+
            "\n"+
            evidence_text
        )

        for triple in triples:
            result=self.verifier.verify(
                triple,
                verification_text
            )

            if result.accepted:
                triple.confidence=min(
                    triple.confidence,
                    result.confidence
                )
                verified.append(triple)

        if not verified:
            return FeedbackResult(
                answer=answer,
                confidence=confidence,
                consistent=True,
                evolved=False,
                triples_added=0,
                citations=self._extract_citations(
                    answer
                ),
                reason=(
                    "Answer was confident, but no "
                    "candidate triple passed verification."
                )
            )

        merged_result=self.merger.merge_triples(
            verified
        )

        if isinstance(
            merged_result,
            tuple
        ):
            merged=merged_result[0]
        else:
            merged=merged_result

        added=0

        for triple in merged:
            graph_triple=Triple(
                subject=triple.subject,
                predicate=triple.predicate,
                object=triple.object,
                passage_id=triple.passage_id,
                confidence=triple.confidence,
                timestamp=time.time()
            )

            committed=self.kg.commit(
                graph_triple
            )

            if committed:
                added+=1

        return FeedbackResult(
            answer=answer,
            confidence=confidence,
            consistent=True,
            evolved=added>0,
            triples_added=added,
            citations=self._extract_citations(
                answer
            ),
            reason=(
                "High-confidence answer verified and "
                "candidate knowledge committed to KG."
                if added>0
                else
                "Verified candidate knowledge did not "
                "meet the KG commit threshold."
            )
        )

    def _extract_candidates(
        self,
        answer,
        evidence
    ):
        passage_id=self._primary_passage_id(
            evidence
        )

        return self.extractor.extract(
            answer,
            passage_id
        )

    def _primary_passage_id(
        self,
        evidence
    ):
        for item in evidence:
            if not isinstance(item,dict):
                continue

            value=(
                item.get("passage_id")
                or item.get("path_id")
            )

            if value:
                return str(value)

        return "generated_answer"

    def _evidence_text(
        self,
        evidence
    ):
        parts=[]

        for item in evidence:
            if not isinstance(item,dict):
                continue

            if item.get("text"):
                parts.append(
                    str(item["text"])
                )

            if item.get("path"):
                parts.append(
                    " -> ".join(
                        str(x)
                        for x in item["path"]
                    )
                )

        return "\n".join(parts)

    def _consistency_score(
        self,
        answers
    ):
        if not answers:
            return 0.0

        if len(answers)==1:
            return (
                1.0
                if answers[0].strip()
                else
                0.0
            )

        scores=[]

        for i in range(len(answers)):
            for j in range(
                i+1,
                len(answers)
            ):
                scores.append(
                    self._token_f1(
                        answers[i],
                        answers[j]
                    )
                )

        if not scores:
            return 0.0

        return (
            sum(scores)/
            len(scores)
        )

    def _token_f1(
        self,
        a,
        b
    ):
        a=self._normalize(a)
        b=self._normalize(b)

        if not a or not b:
            return 0.0

        aa=a.split()
        bb=b.split()

        if not aa or not bb:
            return 0.0

        counts={}

        for token in aa:
            counts[token]=(
                counts.get(
                    token,
                    0
                )+
                1
            )

        overlap=0

        for token in bb:
            if counts.get(
                token,
                0
            )>0:
                overlap+=1
                counts[token]-=1

        if overlap==0:
            return 0.0

        precision=overlap/len(bb)
        recall=overlap/len(aa)

        return (
            2*
            precision*
            recall/
            (precision+recall)
        )

    def _normalize(
        self,
        text
    ):
        text=str(text).lower()

        text=re.sub(
            r"\[[^\]]+\]",
            "",
            text
        )

        text=re.sub(
            r"[^a-z0-9\s]",
            " ",
            text
        )

        text=re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    def _select_answer(
        self,
        answers
    ):
        if not answers:
            return ""

        best=answers[0]
        best_score=-1.0

        for answer in answers:
            score=0.0

            for other in answers:
                score+=self._token_f1(
                    answer,
                    other
                )

            if score>best_score:
                best_score=score
                best=answer

        return best

    def _apply_user_feedback(
        self,
        confidence,
        feedback
    ):
        if isinstance(
            feedback,
            bool
        ):
            feedback_score=(
                1.0
                if feedback
                else
                0.0
            )

        elif isinstance(
            feedback,
            (int,float)
        ):
            feedback_score=max(
                0.0,
                min(
                    1.0,
                    float(feedback)
                )
            )

        else:
            feedback_score=0.0

        w=self.user_feedback_weight

        return (
            (1-w)*
            confidence+
            w*
            feedback_score
        )

    def _extract_citations(
        self,
        answer
    ):
        return re.findall(
            r"\[([^\]]+)\]",
            str(answer)
        )