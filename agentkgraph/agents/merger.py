import re
from difflib import SequenceMatcher


class EntityMerger:
    def __init__(self,threshold=0.92):
        self.threshold=threshold

    def canonicalize(self,name):
        name=re.sub(r"\s+"," ",name.strip())
        return name

    def similarity(self,a,b):
        return SequenceMatcher(
            None,
            a.lower(),
            b.lower()
        ).ratio()

    def resolve(self,name,existing_entities):
        name=self.canonicalize(name)

        if name in existing_entities:
            return name

        best=None
        best_score=0.0

        for entity in existing_entities:
            score=self.similarity(name,entity)

            if score>best_score:
                best_score=score
                best=entity

        if best is not None and best_score>=self.threshold:
            return best

        return name

    def merge_triple(self,triple,existing_entities):
        triple.subject=self.resolve(
            triple.subject,
            existing_entities
        )

        existing_entities.add(triple.subject)

        triple.object=self.resolve(
            triple.object,
            existing_entities
        )

        existing_entities.add(triple.object)

        return triple