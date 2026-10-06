import re
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)


class Synthesizer:
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-3B-Instruct",
        dry_run=False,
        max_new_tokens=128,
        temperature=0.2
    ):
        self.model_name=model_name
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

        quant_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        self.model=AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map="auto"
        )

        self.model.eval()

    def _format_path_node(self,node):
        if isinstance(node,tuple):
            return " | ".join(
                str(x)
                for x in node
            )

        return str(node)

    def _format_evidence(self,evidence):
        parts=[]

        for i,item in enumerate(evidence):
            if not isinstance(item,dict):
                continue

            evidence_id=(
                item.get("path_id")
                or item.get("passage_id")
                or f"evidence_{i+1}"
            )

            if "path" in item:
                path=item["path"]

                path_text=" -> ".join(
                    self._format_path_node(x)
                    for x in path
                )

                parts.append(
                    f"[{evidence_id}] KG path: {path_text}"
                )

            elif "text" in item:
                parts.append(
                    f"[{evidence_id}] Passage: {item['text']}"
                )

            elif "evidence" in item:
                parts.append(
                    f"[{evidence_id}] "
                    f"{str(item['evidence'])}"
                )

        return "\n".join(parts)

    def _build_prompt(self,query,evidence):
        evidence_text=self._format_evidence(
            evidence
        )

        return f"""
Answer the user's question using only the supplied evidence.

Rules:
1. Do not invent facts.
2. If the evidence is insufficient, say so.
3. Give a concise direct answer.
4. Cite KG evidence using its exact [kg_path_xxxx] ID.
5. Cite textual evidence using its exact [passage_id] ID.
6. Every factual claim must be supported by evidence.

Question:
{query}

Evidence:
{evidence_text}

Answer:
""".strip()

    def generate(self,query,evidence):
        if self.dry_run:
            return self._dry_run(
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
                "content":
                    "You are an evidence-grounded "
                    "knowledge graph question answering system."
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

        answer=self.tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()

        return self._clean_answer(
            answer
        )

    def _clean_answer(self,answer):
        answer=re.sub(
            r"^\s*Answer:\s*",
            "",
            answer,
            flags=re.IGNORECASE
        )

        return answer.strip()

    def extract_citations(self,answer):
        return re.findall(
            r"\[(kg_path_\d{4}|[^\]]+)\]",
            str(answer)
        )

    def _dry_run(self,query,evidence):
        if not evidence:
            return {
                "answer":"No sufficient evidence found.",
                "citations":[]
            }

        citations=[]

        for i,item in enumerate(evidence[:4]):
            if not isinstance(item,dict):
                continue

            evidence_id=(
                item.get("path_id")
                or item.get("passage_id")
                or f"evidence_{i+1}"
            )

            citations.append(
                evidence_id
            )

        return {
            "answer":"Evidence retrieved; synthesis required.",
            "citations":citations
        }