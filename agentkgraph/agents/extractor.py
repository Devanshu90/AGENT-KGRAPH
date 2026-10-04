import re
from dataclasses import dataclass


@dataclass
class ExtractedTriple:
    subject:str
    predicate:str
    object:str
    passage_id:str
    confidence:float


class Extractor:
    def __init__(self,dry_run=True):
        self.dry_run=dry_run

    def extract(self,text,passage_id):
        if self.dry_run:
            return self._dry_extract(text,passage_id)

        raise NotImplementedError("Real Qwen extractor will be added later.")

    def _dry_extract(self,text,passage_id):
        triples=[]

        patterns=[
            (
                r"(.+?)\s+(?:directed|directs)\s+(.+?)(?:\.|$)",
                "directed"
            ),
            (
                r"(.+?)\s+(?:was born in|born in)\s+(.+?)(?:\.|$)",
                "born_in"
            ),
            (
                r"(.+?)\s+(?:was released in|released in)\s+(.+?)(?:\.|$)",
                "released_in"
            ),
            (
                r"(.+?)\s+(?:is a|is an)\s+(.+?)(?:\.|$)",
                "is_a"
            ),
            (
                r"(.+?)\s+(?:acted in|starred in)\s+(.+?)(?:\.|$)",
                "acted_in"
            )
        ]

        for pattern,predicate in patterns:
            match=re.search(pattern,text,re.IGNORECASE)

            if match:
                subject=match.group(1).strip()
                obj=match.group(2).strip()

                triples.append(
                    ExtractedTriple(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        passage_id=passage_id,
                        confidence=0.90
                    )
                )

        return triples