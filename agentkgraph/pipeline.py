import gc
import json
import os
import subprocess
import sys
import tempfile

import torch

from agentkgraph.config import Config
from agentkgraph.kg.graph_store import GraphStore
from agentkgraph.routing.router import Router
from agentkgraph.retrieval.kg_paths import KGRetriever
from agentkgraph.retrieval.vector import VectorRetriever
from agentkgraph.retrieval.hybrid import HybridRetriever


class Engine:
    def __init__(
        self,
        passages=None,
        config=None,
        dry_run=False
    ):
        self.config=config or Config()
        self.dry_run=dry_run

        self.kg=GraphStore(
            self.config.kg
        )

        self.router=Router(
            self.config.routing
        )

        self.vector=VectorRetriever(
            passages or [],
            self.config.retrieval.top_k
        )

        self.kg_retriever=None
        self.hybrid=None

        self.extractor=None
        self.verifier=None
        self.merger=None
        self.synthesizer=None
        self.feedback=None

        self._current_kg_path=None

    def load_kg(self,path):
        self._current_kg_path=path

        self.kg.load(path)

        self.kg_retriever=KGRetriever(
            self.kg,
            self.config.kg.max_seeds
        )

        self.hybrid=HybridRetriever(
            self.vector,
            self.kg_retriever,
            self.config.retrieval.hybrid_alpha
        )

    def _load_synthesizer(self):
        if self.synthesizer is not None:
            model=getattr(
                self.synthesizer,
                "model",
                None
            )

            if model is not None:
                return

        from agentkgraph.generation.synthesizer import Synthesizer

        self.synthesizer=Synthesizer(
            dry_run=self.dry_run,
            max_new_tokens=128,
            temperature=0.2
        )

    def _load_evolution_components(self):
        if self.feedback is not None:
            return

        from agentkgraph.agents.extractor import Extractor
        from agentkgraph.agents.verifier import Verifier
        from agentkgraph.agents.merger import EntityMerger
        from agentkgraph.agents.feedback import FeedbackEngine

        self.extractor=Extractor(
            dry_run=self.dry_run
        )

        self.verifier=Verifier(
            threshold=0.60,
            dry_run=self.dry_run
        )

        self.merger=EntityMerger(
            threshold=0.92
        )

        self.feedback=FeedbackEngine(
            synthesizer=None,
            extractor=self.extractor,
            verifier=self.verifier,
            merger=self.merger,
            kg=self.kg,
            consistency_n=2,
            threshold=0.80,
            user_feedback_weight=0.30,
            max_retry=2
        )

    def _free_synthesizer(self):
        if self.synthesizer is None:
            return

        unload=getattr(
            self.synthesizer,
            "unload",
            None
        )

        if unload is not None:
            unload()

        self.synthesizer=None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _free_evolution_models(self):
        if self.extractor is not None:
            unload=getattr(
                self.extractor,
                "unload",
                None
            )

            if unload is not None:
                unload()

        self.extractor=None
        self.verifier=None
        self.merger=None
        self.feedback=None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def retrieve(self,query):
        if self.kg_retriever is None:
            raise RuntimeError(
                "KG not loaded. Call load_kg() first."
            )

        action=self.router.route(
            query
        )

        if action=="kg":
            evidence=self.kg_retriever.search(
                query
            )

        elif action=="vector":
            evidence=self.vector.search(
                query
            )

        else:
            evidence=self.hybrid.search(
                query
            )

        return {
            "query":query,
            "route":action,
            "evidence":evidence
        }

    def _synthesize_two_answers(
        self,
        query,
        evidence
    ):
        self._load_synthesizer()

        first=self.synthesizer.generate(
            query,
            evidence
        )

        second=self.synthesizer.generate(
            query,
            evidence
        )

        if isinstance(first,dict):
            first_answer=first.get(
                "answer",
                ""
            )
        else:
            first_answer=str(first)

        if isinstance(second,dict):
            second_answer=second.get(
                "answer",
                ""
            )
        else:
            second_answer=str(second)

        return first_answer,second_answer

    def _run_evolution_process(
        self,
        query,
        evidence,
        first_answer,
        second_answer,
        user_feedback=None
    ):
        payload={
            "query":query,
            "evidence":evidence,
            "first_answer":first_answer,
            "second_answer":second_answer,
            "user_feedback":user_feedback
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8"
        ) as f:
            json.dump(
                payload,
                f,
                ensure_ascii=False
            )
            payload_path=f.name

        script="""
import json
import sys

from agentkgraph.config import Config
from agentkgraph.kg.graph_store import GraphStore
from agentkgraph.agents.extractor import Extractor
from agentkgraph.agents.verifier import Verifier
from agentkgraph.agents.merger import EntityMerger


def token_f1(a,b):
    a=a.lower().split()
    b=b.lower().split()

    if not a or not b:
        return 0.0

    ca={}
    cb={}

    for x in a:
        ca[x]=ca.get(x,0)+1

    for x in b:
        cb[x]=cb.get(x,0)+1

    common=0

    for x in ca:
        common+=min(
            ca.get(x,0),
            cb.get(x,0)
        )

    if common==0:
        return 0.0

    precision=common/len(a)
    recall=common/len(b)

    if precision+recall==0:
        return 0.0

    return 2*precision*recall/(precision+recall)


def main():
    payload_path=sys.argv[1]
    kg_path=sys.argv[2]

    with open(
        payload_path,
        "r",
        encoding="utf-8"
    ) as f:
        payload=json.load(f)

    query=payload["query"]
    evidence=payload["evidence"]

    answers=[
        payload["first_answer"],
        payload["second_answer"]
    ]

    user_feedback=payload.get(
        "user_feedback"
    )

    consistency=token_f1(
        answers[0],
        answers[1]
    )

    confidence=consistency

    if user_feedback is not None:
        if user_feedback:
            confidence=(
                confidence*0.70+
                0.30
            )
        else:
            confidence=(
                confidence*0.70
            )

    selected=answers[0]

    if len(answers[1].strip())>len(
        answers[0].strip()
    ):
        selected=answers[1]

    result={
        "answer":selected,
        "confidence":float(confidence),
        "consistent":consistency>=0.80,
        "evolved":False,
        "triples_added":0,
        "citations":[],
        "reason":""
    }

    if confidence<0.80:
        result["reason"]=(
            "Confidence below evolution threshold."
        )

        print(
            json.dumps(
                result
            )
        )

        return

    kg=GraphStore(
        Config().kg
    )

    kg.load(
        kg_path
    )

    extractor=Extractor(
        dry_run=False
    )

    verifier=Verifier(
        threshold=0.60,
        dry_run=False
    )

    merger=EntityMerger(
        threshold=0.92
    )

    candidates=extractor.extract(
        selected,
        "evolution_answer"
    )

    added=0

    for triple in candidates:
        triple_dict={
            "subject":triple.subject,
            "predicate":triple.predicate,
            "object":triple.object,
            "passage_id":triple.passage_id,
            "confidence":triple.confidence
        }

        verification=verifier.verify(
            triple_dict,
            selected,
            evidence
        )

        if not verification.accepted:
            continue

        merged=merger.merge_triple(
            triple_dict,
            kg
        )

        if merged is None:
            continue

        committed=kg.commit(
            merged
        )

        if committed:
            added+=1

    kg.save(
        kg_path
    )

    result["evolved"]=added>0
    result["triples_added"]=added
    result["reason"]=(
        "High-confidence answer processed for KG evolution."
    )

    extractor.unload()

    print(
        json.dumps(
            result
        )
    )


if __name__=="__main__":
    main()
"""

        script_path=None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8"
            ) as f:
                f.write(script)
                script_path=f.name

            project_root=os.path.abspath(
                os.getcwd()
            )

            env=os.environ.copy()

            old_pythonpath=env.get(
                "PYTHONPATH",
                ""
            )

            if old_pythonpath:
                env["PYTHONPATH"]=(
                    project_root+
                    os.pathsep+
                    old_pythonpath
                )
            else:
                env["PYTHONPATH"]=project_root

            process=subprocess.run(
                [
                    sys.executable,
                    script_path,
                    payload_path,
                    self._current_kg_path
                ],
                capture_output=True,
                text=True,
                env=env,
                cwd=project_root
            )

            if process.returncode!=0:
                raise RuntimeError(
                    process.stderr.strip()
                    or process.stdout.strip()
                    or "Evolution process failed."
                )

            output=process.stdout.strip()

            lines=output.splitlines()

            json_line=None

            for line in reversed(lines):
                line=line.strip()

                if line.startswith("{") and line.endswith("}"):
                    json_line=line
                    break

            if json_line is None:
                raise RuntimeError(
                    "Evolution process returned no JSON result."
                )

            return json.loads(
                json_line
            )

        finally:
            try:
                os.remove(
                    payload_path
                )
            except OSError:
                pass

            if script_path is not None:
                try:
                    os.remove(
                        script_path
                    )
                except OSError:
                    pass

    def answer(
        self,
        query,
        user_feedback=None,
        evolve=True
    ):
        result=self.retrieve(
            query
        )

        if not evolve:
            self._load_synthesizer()

            generated=self.synthesizer.generate(
                query,
                result["evidence"]
            )

            if isinstance(generated,dict):
                answer=generated.get(
                    "answer",
                    ""
                )

                citations=generated.get(
                    "citations",
                    []
                )

            else:
                answer=str(generated)

                citations=self.synthesizer.extract_citations(
                    answer
                )

            return {
                "query":query,
                "route":result["route"],
                "evidence":result["evidence"],
                "answer":answer,
                "confidence":None,
                "consistent":None,
                "evolved":False,
                "triples_added":0,
                "citations":citations,
                "reason":
                    "Answer synthesis only; "
                    "KG evolution disabled."
            }

        if self._current_kg_path is None:
            raise RuntimeError(
                "KG path is required for evolution."
            )

        first_answer,second_answer=(
            self._synthesize_two_answers(
                query,
                result["evidence"]
            )
        )

        self._free_synthesizer()

        evolution_result=(
            self._run_evolution_process(
                query=query,
                evidence=result["evidence"],
                first_answer=first_answer,
                second_answer=second_answer,
                user_feedback=user_feedback
            )
        )

        return {
            "query":query,
            "route":result["route"],
            "evidence":result["evidence"],
            "answer":evolution_result.get(
                "answer",
                first_answer
            ),
            "confidence":evolution_result.get(
                "confidence"
            ),
            "consistent":evolution_result.get(
                "consistent"
            ),
            "evolved":evolution_result.get(
                "evolved",
                False
            ),
            "triples_added":evolution_result.get(
                "triples_added",
                0
            ),
            "citations":evolution_result.get(
                "citations",
                []
            ),
            "reason":evolution_result.get(
                "reason",
                ""
            )
        }

    def update_router(
        self,
        query,
        action,
        reward
    ):
        return self.router.update(
            query,
            action,
            reward
        )

    def decay_kg(self):
        return self.kg.decay()

    def save_kg(self,path):
        self.kg.save(path)

    def unload_all(self):
        self._free_synthesizer()
        self._free_evolution_models()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()