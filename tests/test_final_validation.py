from agentkgraph.evaluation.final_validation import (
    FinalValidator,
    run_final_validation
)


def test_final_validator_dry_run():
    validator=FinalValidator(
        dry_run=True
    )

    result=validator.run()

    assert result["success"] is True
    assert result["failed"]==0
    assert result["total"]==10


def test_final_validator_contains_core_checks():
    validator=FinalValidator(
        dry_run=True
    )

    result=validator.run()

    names={
        check["name"]
        for check in result["checks"]
    }

    assert "environment" in names
    assert "configuration" in names
    assert "engine_end_to_end" in names
    assert "ucb_router" in names
    assert "evolution_evaluator" in names
    assert "ablation" in names


def test_final_validation_reports(tmp_path):
    result=run_final_validation(
        dry_run=True,
        output_dir=str(tmp_path)
    )

    assert result["success"] is True
    assert (tmp_path/"final_validation.json").exists()
    assert (tmp_path/"final_validation.txt").exists()