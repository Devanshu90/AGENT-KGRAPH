import json

from agentkgraph.evaluation.real_model_benchmark import (
    BenchmarkResult,
    RealModelBenchmark,
)


def test_benchmark_result_serializable():
    result=BenchmarkResult(
        name="test",
        status="PASS",
        load_success=True,
        inference_success=True,
        load_time_ms=1.0,
        inference_time_ms=2.0,
        peak_memory_kb=3.0,
        details={"output":"ok"},
    )

    data=result.__dict__

    assert data["name"]=="test"
    assert data["status"]=="PASS"
    json.dumps(data)


def test_safe_details():
    benchmark=RealModelBenchmark()

    assert benchmark._safe_details(None)=={}
    assert benchmark._safe_details("ok")["output"]=="ok"
    assert benchmark._safe_details([1,2,3])["output_length"]==3


def test_report_generation(tmp_path):
    benchmark=RealModelBenchmark(
        output_dir=str(tmp_path),
        sample_size=1,
    )

    result={
        "benchmark_mode":"real_model",
        "sample_size":1,
        "real_model_benchmark_status":"completed",
        "success":True,
        "passed":1,
        "loaded":1,
        "inference_successes":1,
        "total":1,
        "total_duration_ms":10.0,
        "environment":{
            "python":"3.13",
            "platform":"test",
            "dry_run":False,
            "cuda_available":False,
        },
        "models":{
            "extractor":"test-extractor",
            "verifier":"test-verifier",
            "router":"test-router",
            "synthesizer":"test-synthesizer",
            "embedder":"test-embedder",
        },
        "results":[
            {
                "name":"test",
                "status":"PASS",
                "load_success":True,
                "inference_success":True,
                "load_time_ms":1.0,
                "inference_time_ms":2.0,
                "peak_memory_kb":3.0,
                "details":{},
                "error":None,
            }
        ],
    }

    paths=benchmark.save_reports(result)

    assert (tmp_path/"real_model_benchmark.json").exists()
    assert (tmp_path/"real_model_benchmark.txt").exists()

    with open(paths["json"],"r",encoding="utf-8") as f:
        saved=json.load(f)

    assert saved["success"] is True
    assert saved["total"]==1
