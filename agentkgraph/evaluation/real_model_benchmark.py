import gc
import json
import platform
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

from agentkgraph.config import Config


@dataclass
class BenchmarkResult:
    name: str
    status: str
    load_success: bool
    inference_success: bool
    load_time_ms: float
    inference_time_ms: float
    peak_memory_kb: float
    details: dict
    error: str | None = None

    def to_dict(self):
        return {
            "name": self.name,
            "status": self.status,
            "load_success": self.load_success,
            "inference_success": self.inference_success,
            "load_time_ms": self.load_time_ms,
            "inference_time_ms": self.inference_time_ms,
            "peak_memory_kb": self.peak_memory_kb,
            "details": self.details,
            "error": self.error
        }


class RealModelBenchmark:
    def __init__(
        self,
        config=None,
        output_dir="reports/b9",
        sample_size=1
    ):
        self.config=config or Config()
        self.output_dir=Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )
        self.sample_size=sample_size

    def _measure(self,name,load_fn,infer_fn):
        model=None
        load_success=False
        inference_success=False
        load_time=0.0
        inference_time=0.0
        peak_memory=0.0
        details={}
        error=None

        gc.collect()
        tracemalloc.start()

        load_start=time.perf_counter()

        try:
            model=load_fn()

            load_time=(
                time.perf_counter()-load_start
            )*1000

            load_success=True

        except Exception as exc:
            load_time=(
                time.perf_counter()-load_start
            )*1000

            error=f"{type(exc).__name__}: {exc}"

        if load_success:
            try:
                tracemalloc.reset_peak()

                inference_start=time.perf_counter()

                output=infer_fn(model)

                inference_time=(
                    time.perf_counter()-inference_start
                )*1000

                inference_success=True
                details=self._safe_details(output)

            except Exception as exc:
                inference_time=(
                    time.perf_counter()-inference_start
                )*1000

                error=f"{type(exc).__name__}: {exc}"

        _,peak=tracemalloc.get_traced_memory()

        peak_memory=peak/1024

        tracemalloc.stop()

        if model is not None:
            try:
                if hasattr(model,"unload"):
                    model.unload()
            except Exception:
                pass

        del model
        gc.collect()

        if not load_success:
            status="LOAD_FAILED"
        elif not inference_success:
            status="INFERENCE_FAILED"
        else:
            status="PASS"

        return BenchmarkResult(
            name=name,
            status=status,
            load_success=load_success,
            inference_success=inference_success,
            load_time_ms=round(load_time,3),
            inference_time_ms=round(inference_time,3),
            peak_memory_kb=round(peak_memory,3),
            details=details,
            error=error
        )

    def _safe_details(self,output):
        if output is None:
            return {}

        if isinstance(
            output,
            (str,int,float,bool)
        ):
            return {
                "output":output
            }

        if isinstance(output,dict):
            result={}

            for key,value in output.items():
                if isinstance(
                    value,
                    (str,int,float,bool,type(None))
                ):
                    result[key]=value
                elif isinstance(
                    value,
                    (list,tuple,set)
                ):
                    result[
                        f"{key}_length"
                    ]=len(value)
                else:
                    result[key]=str(value)

            return result

        if isinstance(output,(list,tuple)):
            result={
                "output_length":len(output)
            }

            if output:
                result[
                    "first_item_type"
                ]=type(
                    output[0]
                ).__name__

            return result

        return {
            "output_type":type(output).__name__
        }

    def benchmark_extractor(self):
        from agentkgraph.agents.extractor import Extractor

        def load():
            return Extractor(
                model_name=self.config.models.extractor,
                adapter_path=self.config.models.extractor_adapter,
                dry_run=False,
                max_new_tokens=self.config.models.extractor_max_new_tokens,
                temperature=self.config.models.extractor_temperature
            )

        def infer(model):
            return model.extract(
                "Alice works at Acme Corporation and lives in Bhopal.",
                "benchmark_passage_001"
            )

        return self._measure(
            "extractor",
            load,
            infer
        )

    def benchmark_verifier(self):
        from agentkgraph.agents.extractor import ExtractedTriple
        from agentkgraph.agents.verifier import Verifier

        def load():
            return Verifier(
                threshold=0.60,
                model_name=self.config.models.verifier,
                dry_run=False
            )

        def infer(model):
            triple=ExtractedTriple(
                subject="Alice",
                predicate="works_at",
                object="Acme Corporation",
                passage_id="benchmark_passage_001",
                confidence=0.95
            )

            return model.verify(
                triple,
                "Alice works at Acme Corporation."
            )

        return self._measure(
            "verifier",
            load,
            infer
        )

    def benchmark_embedder(self):
        from agentkgraph.retrieval.vector import VectorRetriever

        passages=[
            {
                "passage_id":"benchmark_passage_001",
                "document_id":"benchmark_document_001",
                "text":"Alice works at Acme Corporation in Bhopal."
            }
        ]

        def load():
            return VectorRetriever(
                passages=passages,
                top_k=self.config.retrieval.top_k,
                model_name=self.config.models.embedder,
                dry_run=False
            )

        def infer(model):
            return model.search(
                "Where does Alice work?"
            )

        return self._measure(
            "embedder",
            load,
            infer
        )

    def benchmark_router(self):
        from agentkgraph.routing.router import Router

        def load():
            return Router(
                config=self.config.routing,
                model_name=self.config.models.router,
                dry_run=False,
                max_new_tokens=self.config.models.router_max_new_tokens
            )

        def infer(model):
            return model.route(
                "Where does Alice work?"
            )

        return self._measure(
            "router",
            load,
            infer
        )

    def benchmark_synthesizer(self):
        from agentkgraph.generation.synthesizer import Synthesizer

        evidence=[
            {
                "text":"Alice works at Acme Corporation.",
                "passage_id":"benchmark_passage_001",
                "document_id":"benchmark_document_001",
                "route":"vector"
            }
        ]

        def load():
            return Synthesizer(
                model_name=self.config.models.synthesizer,
                dry_run=False,
                max_new_tokens=8,
                temperature=0.0,
                load_in_4bit=True
            )

        def infer(model):
            return model.generate(
                "Where does Alice work?",
                evidence
            )

        return self._measure(
            "synthesizer",
            load,
            infer
        )

    def run_component(self,name):
        components={
            "extractor":self.benchmark_extractor,
            "verifier":self.benchmark_verifier,
            "embedder":self.benchmark_embedder,
            "router":self.benchmark_router,
            "synthesizer":self.benchmark_synthesizer
        }

        if name not in components:
            raise ValueError(
                f"Unknown component: {name}. "
                f"Available: {', '.join(components.keys())}"
            )

        total_start=time.perf_counter()

        result=components[name]()

        total_duration=(
            time.perf_counter()-total_start
        )*1000

        payload={
            "benchmark_mode":"real_model",
            "component":name,
            "sample_size":self.sample_size,
            "real_model_benchmark_status":"completed",
            "success":result.status=="PASS",
            "total_duration_ms":round(
                total_duration,
                3
            ),
            "environment":{
                "python":platform.python_version(),
                "platform":platform.platform(),
                "cuda_available":self._cuda_available()
            },
            "models":{
                "extractor":self.config.models.extractor,
                "verifier":self.config.models.verifier,
                "router":self.config.models.router,
                "synthesizer":self.config.models.synthesizer,
                "embedder":self.config.models.embedder
            },
            "result":result.to_dict()
        }

        paths=self.save_component_report(
            name,
            payload
        )

        payload["reports"]=paths

        print(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False
            )
        )

        return payload

    def save_component_report(self,name,payload):
        json_path=(
            self.output_dir/
            f"real_model_{name}.json"
        )

        text_path=(
            self.output_dir/
            f"real_model_{name}.txt"
        )

        with json_path.open(
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False
            )

        result=payload["result"]

        lines=[
            f"Real Model Benchmark: {name}",
            "="*60,
            f"Status: {result['status']}",
            f"Load success: {result['load_success']}",
            f"Inference success: {result['inference_success']}",
            f"Load time (ms): {result['load_time_ms']}",
            f"Inference time (ms): {result['inference_time_ms']}",
            f"Peak memory (KB): {result['peak_memory_kb']}",
            ""
        ]

        if result.get("error"):
            lines.extend([
                f"Error: {result['error']}",
                ""
            ])

        lines.append("Details:")

        lines.append(
            json.dumps(
                result.get(
                    "details",
                    {}
                ),
                indent=2,
                ensure_ascii=False
            )
        )

        with text_path.open(
            "w",
            encoding="utf-8"
        ) as file:
            file.write(
                "\n".join(lines)
            )

        return {
            "json":str(json_path),
            "text":str(text_path)
        }

    def save_reports(self,result):
        json_path=(
            self.output_dir/
            "real_model_benchmark.json"
        )

        text_path=(
            self.output_dir/
            "real_model_benchmark.txt"
        )

        with json_path.open(
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                result,
                file,
                indent=2,
                ensure_ascii=False
            )

        lines=[
            "Real Model Benchmark",
            "="*60,
            f"Success: {result.get('success',False)}",
            f"Total: {result.get('total',0)}",
            f"Passed: {result.get('passed',0)}",
            f"Loaded: {result.get('loaded',0)}",
            f"Inference successes: {result.get('inference_successes',0)}",
            f"Total duration (ms): {result.get('total_duration_ms',0)}",
            ""
        ]

        for item in result.get("results",[]):
            lines.extend([
                f"Component: {item.get('name','unknown')}",
                f"Status: {item.get('status','unknown')}",
                f"Load success: {item.get('load_success',False)}",
                f"Inference success: {item.get('inference_success',False)}",
                f"Load time (ms): {item.get('load_time_ms',0)}",
                f"Inference time (ms): {item.get('inference_time_ms',0)}",
                f"Peak memory (KB): {item.get('peak_memory_kb',0)}",
                f"Error: {item.get('error')}",
                ""
            ])

        with text_path.open(
            "w",
            encoding="utf-8"
        ) as file:
            file.write(
                "\n".join(lines)
            )

        return {
            "json":str(json_path),
            "text":str(text_path)
        }

    def _cuda_available(self):
        try:
            import torch

            return bool(
                torch.cuda.is_available()
            )

        except Exception:
            return False


def main():
    component="extractor"

    if len(sys.argv)>1:
        component=(
            sys.argv[1]
            .strip()
            .lower()
        )

    benchmark=RealModelBenchmark()

    benchmark.run_component(
        component
    )


if __name__=="__main__":
    main()