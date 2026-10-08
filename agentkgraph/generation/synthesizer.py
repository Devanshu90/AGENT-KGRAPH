import torch
from transformers import AutoModelForCausalLM,AutoTokenizer


class Synthesizer:
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-3B-Instruct",
        dry_run=False,
        max_new_tokens=128,
        temperature=0.2,
        load_in_4bit=False
    ):
        self.model_name=model_name
        self.dry_run=dry_run
        self.max_new_tokens=max_new_tokens
        self.temperature=temperature
        self.load_in_4bit=load_in_4bit
        self.model=None
        self.tokenizer=None
        self.device="cuda" if torch.cuda.is_available() else "cpu"

        if not self.dry_run:
            self._load_model()

    def _load_model(self):
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token=self.tokenizer.eos_token

        kwargs={}

        if torch.cuda.is_available():
            kwargs["torch_dtype"]=torch.float16
            kwargs["device_map"]="auto"

            if self.load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig

                    kwargs["quantization_config"]=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True
                    )
                    kwargs.pop("torch_dtype",None)
                except ImportError as exc:
                    raise ImportError(
                        "bitsandbytes is required for 4-bit loading. "
                        "Install it with: pip install bitsandbytes"
                    ) from exc
        else:
            kwargs["torch_dtype"]=torch.float32

        self.model=AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **kwargs
        )

        if not torch.cuda.is_available():
            self.model.to(self.device)

        self.model.eval()

    def _build_prompt(self,query,evidence):
        evidence_text=[]

        for i,item in enumerate(evidence,1):
            if isinstance(item,dict):
                text=item.get("text","")
                source=item.get("passage_id") or item.get("document_id") or f"evidence_{i}"
                route=item.get("route","")

                if route:
                    evidence_text.append(
                        f"[{i}] source={source} route={route}\n{text}"
                    )
                else:
                    evidence_text.append(
                        f"[{i}] source={source}\n{text}"
                    )
            else:
                evidence_text.append(f"[{i}] {item}")

        context="\n\n".join(evidence_text)

        return f"""
Answer the question using only the provided evidence.

Rules:
- Do not invent facts.
- If the evidence is insufficient, say that the evidence is insufficient.
- Give a concise, direct answer.
- Preserve important entity names exactly when possible.
- Do not mention internal model details.
- Include source identifiers in a final "Sources:" line.

Question:
{query}

Evidence:
{context}

Answer:
""".strip()

    def generate(self,query,evidence):
        if self.dry_run:
            return self._dry_generate(query,evidence)

        prompt=self._build_prompt(query,evidence)

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

        if hasattr(self.model,"device"):
            inputs=inputs.to(self.model.device)
        else:
            inputs=inputs.to(self.device)

        with torch.no_grad():
            outputs=self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature>0,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated=outputs[0][inputs["input_ids"].shape[1]:]

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
            text=str(first.get("text","")).strip()
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
