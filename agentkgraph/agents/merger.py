import re
from difflib import SequenceMatcher
from dataclasses import dataclass


@dataclass
class MergeResult:
    canonical:str
    matched:bool
    score:float


class EntityMerger:
    def __init__(self,threshold=0.92):
        self.threshold=threshold

    def canonicalize(self,name):
        name=str(name).strip()
        name=re.sub(r"\s+"," ",name)
        return name

    def normalize(self,name):
        name=self.canonicalize(name)
        return name.casefold()

    def similarity(self,a,b):
        return SequenceMatcher(
            None,
            self.normalize(a),
            self.normalize(b)
        ).ratio()

    def resolve(self,name,existing_entities):
        name=self.canonicalize(name)

        if not name:
            return MergeResult(
                canonical=name,
                matched=False,
                score=0.0
            )

        normalized=self.normalize(name)

        for entity in existing_entities:
            if self.normalize(entity)==normalized:
                return MergeResult(
                    canonical=entity,
                    matched=True,
                    score=1.0
                )

        best=None
        best_score=0.0

        for entity in existing_entities:
            score=self.similarity(
                name,
                entity
            )

            if score>best_score:
                best_score=score
                best=entity

        if best is not None and best_score>=self.threshold:
            return MergeResult(
                canonical=best,
                matched=True,
                score=best_score
            )

        return MergeResult(
            canonical=name,
            matched=False,
            score=best_score
        )

    def merge_triple(self,triple,existing_entities):
        subject=self.resolve(
            triple.subject,
            existing_entities
        )

        triple.subject=subject.canonical
        existing_entities.add(triple.subject)

        obj=self.resolve(
            triple.object,
            existing_entities
        )

        triple.object=obj.canonical
        existing_entities.add(triple.object)

        return triple

    def merge_entities(self,names):
        existing=set()
        mapping={}

        for name in names:
            result=self.resolve(
                name,
                existing
            )

            mapping[name]=result.canonical
            existing.add(result.canonical)

        return mapping

    def merge_triples(self,triples,existing_entities=None):
        entities=existing_entities or set()
        merged=[]

        for triple in triples:
            merged.append(
                self.merge_triple(
                    triple,
                    entities
                )
            )

        return merged,entities