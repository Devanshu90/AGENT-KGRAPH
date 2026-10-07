import json

from agentkgraph.evaluation.report import EvaluationReport


def test_report_creation():
    report=EvaluationReport()

    report.add_section(
        "Overall Metrics",
        {
            "exact_match":0.80,
            "token_f1":0.85,
        },
    )

    result=report.to_dict()

    assert result["title"]=="AGENT-KGRAPH Evaluation Report"
    assert len(result["sections"])==1
    assert result["sections"][0]["name"]=="Overall Metrics"


def test_multiple_sections():
    report=EvaluationReport()

    report.add_section(
        "Metrics",
        {"exact_match":0.8},
    )

    report.add_section(
        "Routing",
        {"accuracy":0.9},
    )

    result=report.to_dict()

    assert len(result["sections"])==2
    assert result["sections"][1]["data"]["accuracy"]==0.9


def test_save_json(tmp_path):
    report=EvaluationReport()

    report.add_section(
        "Metrics",
        {"exact_match":0.8},
    )

    path=tmp_path/"report.json"
    saved=report.save_json(path)

    assert saved==path
    assert path.exists()

    with path.open("r",encoding="utf-8") as f:
        data=json.load(f)

    assert data["sections"][0]["data"]["exact_match"]==0.8


def test_text_report(tmp_path):
    report=EvaluationReport()

    report.add_section(
        "Metrics",
        {
            "exact_match":0.8,
            "token_f1":0.85,
        },
    )

    text=report.to_text()

    assert "AGENT-KGRAPH Evaluation Report" in text
    assert "Metrics" in text
    assert "exact_match: 0.8" in text


def test_save_text(tmp_path):
    report=EvaluationReport()

    report.add_section(
        "Routing",
        {"accuracy":0.9},
    )

    path=tmp_path/"report.txt"
    saved=report.save_text(path)

    assert saved==path
    assert path.exists()

    content=path.read_text(
        encoding="utf-8"
    )

    assert "Routing" in content
    assert "accuracy: 0.9" in content