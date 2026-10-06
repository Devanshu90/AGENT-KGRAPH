import json
import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer


@dataclass
class ExtractedTriple:
    subject:str
    predicate:str
    object:str
    passage_id:str
    confidence:float


class Extractor:
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path=None,
        dry_run=False,
        max_new_tokens=256,
        temperature=0.1
    ):
        self.model_name=model_name
        self.adapter_path=adapter_path
        self.dry_run=dry_run
        self.max_new_tokens=max_new_tokens
        self.temperature=temperature
        self.model=None
        self.tokenizer=None

        if not self.dry_run:
            self._load_model()

    def _load_model(self):
        self.tokenizer=AutoTokenizer.from_pretrained(
            self.model_name
        )

        dtype=(
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        )

        self.model=AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map="auto"
        )

        if self.adapter_path:
            from peft import PeftModel

            self.model=PeftModel.from_pretrained(
                self.model,
                self.adapter_path
            )

        self.model.eval()

    def _build_prompt(self,text):
        return f"""
Extract factual knowledge from the passage.

Return ONLY a JSON array.

Each item must contain:
subject
predicate
object
confidence

Use normalized predicates such as:
directed_by
written_by
starred_actors
has_genre
release_year
born_in
married_to

Passage:
{text}

JSON:
""".strip()

    def extract(self,text,passage_id):
        if self.dry_run:
            return self._dry_extract(
                text,
                passage_id
            )

        prompt=self._build_prompt(
            text
        )

        messages=[
            {
                "role":"system",
                "content":
                    "You are a factual knowledge extraction system."
            },
            {
                "role":"user",
                "content":prompt
            }
        ]

        formatted=self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs=self.tokenizer(
            formatted,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs=self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature>0,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated=outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        response=self.tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()

        return self._parse(
            response,
            passage_id
        )

    def _parse(self,response,passage_id):
        match=re.search(
            r"\[[\s\S]*\]",
            response
        )

        if not match:
            return []

        try:
            data=json.loads(
                match.group(0)
            )
        except json.JSONDecodeError:
            return []

        triples=[]

        for item in data:
            if not isinstance(item,dict):
                continue

            subject=str(
                item.get("subject","")
            ).strip()

            predicate=str(
                item.get("predicate","")
            ).strip()

            obj=str(
                item.get("object","")
            ).strip()

            if not subject or not predicate or not obj:
                continue

            try:
                confidence=float(
                    item.get(
                        "confidence",
                        0.5
                    )
                )
            except (TypeError,ValueError):
                confidence=0.5

            confidence=max(
                0.0,
                min(
                    1.0,
                    confidence
                )
            )

            triples.append(
                ExtractedTriple(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    passage_id=passage_id,
                    confidence=confidence
                )
            )

        return triples

    def _dry_extract(
        self,
        text,
        passage_id
    ):
        text=text.strip()

        match=re.search(
            r"(.+?)\s+was\s+directed\s+by\s+(.+?)[.!]?$",
            text,
            flags=re.IGNORECASE
        )

        if match:
            return [
                ExtractedTriple(
                    subject=match.group(1).strip(),
                    predicate="directed_by",
                    object=match.group(2).strip(),
                    passage_id=passage_id,
                    confidence=0.95
                )
            ]

        return []

    def to_dicts(self,triples):
        return [
            {
                "subject":t.subject,
                "predicate":t.predicate,
                "object":t.object,
                "passage_id":t.passage_id,
                "confidence":t.confidence
            }
            for t in triples
        ]

    def unload(self):
        if self.model is not None:
            del self.model
            self.model=None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer=None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()