import json
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer


class Router:
    def __init__(
        self,
        config=None,
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        dry_run=False
    ):
        self.config=config

        self.actions=[
            "vector",
            "kg",
            "hybrid"
        ]

        self.beta=1.0
        self.min_explore=30

        if config is not None:
            self.beta=getattr(
                config,
                "ucb_beta",
                self.beta
            )

            self.min_explore=getattr(
                config,
                "min_explore",
                self.min_explore
            )

        self.counts={
            action:0
            for action in self.actions
        }

        self.dimension=8

        self.A={
            action:np.eye(self.dimension)
            for action in self.actions
        }

        self.b={
            action:np.zeros(self.dimension)
            for action in self.actions
        }

        self.dry_run=dry_run
        self.model_name=model_name
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

        self.model.eval()

    def features(self,query):
        q=query.lower()

        return {
            "has_entity":bool(
                re.search(
                    r"\[[^\]]+\]",
                    query
                )
            ),
            "has_who":q.startswith("who"),
            "has_what":q.startswith("what"),
            "has_where":q.startswith("where"),
            "has_when":q.startswith("when"),
            "has_how":q.startswith("how"),
            "length":len(q.split())
        }

    def _context(self,query):
        f=self.features(query)

        length=min(
            f["length"]/20.0,
            1.0
        )

        return np.array([
            float(f["has_entity"]),
            float(f["has_who"]),
            float(f["has_what"]),
            float(f["has_where"]),
            float(f["has_when"]),
            float(f["has_how"]),
            length,
            1.0
        ],dtype=np.float64)

    def _prompt(self,query):
        return f"""
You are a retrieval routing classifier.

Choose the best retrieval strategy for the query.

Available strategies:
vector = semantic text/passages
kg = knowledge graph traversal
hybrid = both vector and knowledge graph

Return JSON only in this format:
{{"vector":0.0,"kg":0.0,"hybrid":0.0}}

The three values must be probabilities between 0 and 1
and should sum approximately to 1.

Query:
{query}
""".strip()

    def _model_scores(self,query):
        if self.dry_run:
            return self._dry_scores(query)

        messages=[
            {
                "role":"system",
                "content":
                    "You are a retrieval routing classifier."
            },
            {
                "role":"user",
                "content":self._prompt(query)
            }
        ]

        prompt=self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs=self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs=self.model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated=outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        response=self.tokenizer.decode(
            generated,
            skip_special_tokens=True
        )

        match=re.search(
            r"\{[\s\S]*\}",
            response
        )

        if not match:
            return self._dry_scores(query)

        try:
            data=json.loads(
                match.group(0)
            )
        except json.JSONDecodeError:
            return self._dry_scores(query)

        scores=np.array([
            float(data.get("vector",0.0)),
            float(data.get("kg",0.0)),
            float(data.get("hybrid",0.0))
        ])

        scores=np.maximum(
            scores,
            0.0
        )

        total=scores.sum()

        if total<=0:
            return self._dry_scores(query)

        scores=scores/total

        return {
            action:float(scores[i])
            for i,action in enumerate(self.actions)
        }

    def _dry_scores(self,query):
        f=self.features(query)

        scores={
            "vector":0.34,
            "kg":0.33,
            "hybrid":0.33
        }

        if f["has_entity"]:
            scores={
                "vector":0.10,
                "kg":0.75,
                "hybrid":0.15
            }

        elif (
            f["has_who"]
            and "direct" in query.lower()
        ):
            scores={
                "vector":0.15,
                "kg":0.20,
                "hybrid":0.65
            }

        elif f["has_when"]:
            scores={
                "vector":0.20,
                "kg":0.20,
                "hybrid":0.60
            }

        return scores

    def _ucb_scores(self,context,prior):
        scores={}

        total_count=sum(
            self.counts.values()
        )

        for action in self.actions:
            A_inv=np.linalg.inv(
                self.A[action]
            )

            theta=A_inv@self.b[action]

            mean=float(
                theta@context
            )

            uncertainty=float(
                np.sqrt(
                    context@A_inv@context
                )
            )

            exploration=(
                self.beta*uncertainty
            )

            if total_count<self.min_explore:
                exploration*=0.25

            scores[action]=(
                mean
                + exploration
                + prior[action]
            )

        return scores

    def _obvious_route(self,query):
        f=self.features(query)

        if f["has_entity"]:
            return "kg"

        q=query.lower()

        if (
            f["has_who"]
            and "direct" in q
        ):
            return "hybrid"

        if f["has_when"]:
            return "hybrid"

        return None

    def route(self,query):
        obvious=self._obvious_route(
            query
        )

        if obvious is not None:
            return obvious

        context=self._context(query)
        prior=self._model_scores(query)

        scores=self._ucb_scores(
            context,
            prior
        )

        return max(
            self.actions,
            key=lambda a:scores[a]
        )

    def route_with_scores(self,query):
        context=self._context(query)
        prior=self._model_scores(query)

        scores=self._ucb_scores(
            context,
            prior
        )

        obvious=self._obvious_route(
            query
        )

        action=(
            obvious
            if obvious is not None
            else max(
                self.actions,
                key=lambda a:scores[a]
            )
        )

        return {
            "action":action,
            "probabilities":prior,
            "ucb_scores":{
                k:float(v)
                for k,v in scores.items()
            }
        }

    def update(self,query,action,reward):
        if action not in self.actions:
            raise ValueError(
                f"Unknown routing action: {action}"
            )

        context=self._context(query)

        self.A[action]+=np.outer(
            context,
            context
        )

        self.b[action]+=(
            float(reward)*context
        )

        self.counts[action]+=1

    def reset(self):
        self.counts={
            action:0
            for action in self.actions
        }

        self.A={
            action:np.eye(self.dimension)
            for action in self.actions
        }

        self.b={
            action:np.zeros(self.dimension)
            for action in self.actions
        }