import gc

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer


class Synthesizer:
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-3B-Instruct",
        dry_run=False,
        max_new_tokens=128,
        temperature=0.2,
        load_in_4bit=True
    ):
        self.model_name=model_name
        self.dry_run=dry_run
        self.max_new_tokens=max_new_tokens
        self.temperature=temperature
        self.load_in_4bit=load_in_4bit
        self.model=None
        self.tokenizer=None
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.dtype=None

        if not self.dry_run:
            self._load_model()

    def _load_model(self):
        self.tokenizer=AutoTokenizer.from_pretrained(
            self.model_name
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token=self.tokenizer.eos_token

        if self.load_in_4bit:
            self._load_quantized()
        else:
            self._load_standard()

        self.model.eval()

    def _load_quantized(self):
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "bitsandbytes and BitsAndBytesConfig are required for 4-bit model loading."
            ) from exc

        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True
        )

        kwargs={
            "quantization_config":quantization_config,
            "low_cpu_mem_usage":True
        }

        if torch.cuda.is_available():
            kwargs["device_map"]="auto"
        else:
            kwargs["device_map"]={"":"cpu"}

        self.model=AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **kwargs
        )

        self.device="cuda" if torch.cuda.is_available() else "cpu"

    def _load_standard(self):
        kwargs={
            "low_cpu_mem_usage":True
        }

        if torch.cuda.is_available():
            self.dtype=torch.float16
            kwargs["dtype"]=torch.float16
            kwargs["device_map"]="auto"
        else:
            self.dtype=torch.float16
            kwargs["dtype"]=torch.float16

        self.model=AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **kwargs
        )

        if not torch.cuda.is_available():
            self.model=self.model.to(self.device)

    def _format_path(self,path):
        if not path:
            return ""

        parts=[]

        for item in path:
            if isinstance(item,(list,tuple)):
                if len(item)==2:
                    predicate,node=item

                    if predicate is None:
                        parts.append(str(node))
                    else:
                        parts.append(
                            f"{predicate} -> {node}"
                        )
                else:
                    parts.append(
                        " -> ".join(str(x) for x in item)
                    )
            else:
                parts.append(str(item))

        return " -> ".join(parts)

    def _build_prompt(self,query,evidence):
        evidence_text=[]

        for i,item in enumerate(evidence,1):
            if isinstance(item,dict):
                path=item.get("path")

                if path:
                    path_text=self._format_path(path)

                    source=(
                        item.get("path_id")
                        or item.get("passage_id")
                        or item.get("document_id")
                        or f"evidence_{i}"
                    )

                    score=item.get("score","")
                    seed=item.get("seed","")

                    evidence_text.append(
                        f"[{i}] KG PATH\n"
                        f"source={source}\n"
                        f"seed={seed}\n"
                        f"path={path_text}\n"
                        f"confidence={score}"
                    )

                    continue

                text=item.get("text","")

                source=(
                    item.get("passage_id")
                    or item.get("document_id")
                    or f"evidence_{i}"
                )

                route=item.get("route","")

                if route:
                    evidence_text.append(
                        f"[{i}] source={source} route={route}\n"
                        f"{text}"
                    )
                else:
                    evidence_text.append(
                        f"[{i}] source={source}\n"
                        f"{text}"
                    )

            else:
                evidence_text.append(
                    f"[{i}] {item}"
                )

        context="\n\n".join(evidence_text)

        return f"""
You are a grounded multi-hop question answering system.

Answer the question using ONLY the provided evidence.

IMPORTANT REASONING RULES:

1. Read every KG PATH from left to right.

2. A KG path has the form:
   entity -> relation -> entity

3. Relations marked "(reverse)" mean that the graph edge is being followed backwards.

4. For multi-hop questions, follow the complete chain.

5. Identify the relation that directly answers the question.

6. Return the entity at the END of the relation that answers the question.

7. Ignore paths whose final relation does not answer the question.

8. When multiple valid paths answer the question, return ALL unique answers.

9. Do not return intermediate movie/entity names unless the question asks for them.

10. Do not invent facts that are not present in the evidence.

11. Give only the direct answer, followed by a Sources line.

Example:

Question:
which people directed movies starred by [Person]

Path:
Person -> starred_actors (reverse) -> Movie -> directed_by -> Director

Answer:
Director

If several such paths exist:

Answer:
Director 1, Director 2, Director 3

For the current question, determine the requested relation from the wording
of the question.

Prefer the relation whose meaning directly answers the question over relations
that only help reach the answer.

Question:
{query}

Evidence:
{context}

Answer:
""".strip()

    def generate(self,query,evidence):
        if self.dry_run:
            return self._dry_generate(
                query,
                evidence
            )

        prompt=self._build_prompt(
            query,
            evidence
        )

        messages=[
            {
                "role":"system",
                "content":"You are a grounded question-answering system."
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
            return_tensors="pt",
            truncation=True
        )

        inputs=inputs.to(self.device)

        generation_kwargs={
            "max_new_tokens":self.max_new_tokens,
            "do_sample":self.temperature>0,
            "pad_token_id":self.tokenizer.pad_token_id
        }

        if self.temperature>0:
            generation_kwargs["temperature"]=self.temperature

        with torch.no_grad():
            outputs=self.model.generate(
                **inputs,
                **generation_kwargs
            )

        generated=outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        response=self.tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()

        return response

    def _dry_generate(self,query,evidence):
        if not evidence:
            return {
                "answer":"The evidence is insufficient to answer the question.",
                "sources":[]
            }

        first=evidence[0]

        if isinstance(first,dict):
            if first.get("path"):
                path=self._format_path(
                    first["path"]
                )

                source=first.get(
                    "path_id",
                    "unknown"
                )

                return {
                    "answer":path,
                    "sources":[source]
                }

            text=str(
                first.get("text","")
            ).strip()

            source=(
                first.get("passage_id")
                or first.get("document_id")
                or "unknown"
            )

        else:
            text=str(first).strip()
            source="unknown"

        return {
            "answer":text,
            "sources":[source]
        }

    def unload(self):
        if self.model is not None:
            del self.model
            self.model=None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer=None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()