import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import wait_and_finalize_confirmatory as watcher


PREREQUISITES = [
    "cap_sweep_summary.json",
    "stability_summary.json",
    "final_cap_decisions.json",
    "TPE_SUITE_COMPLETE",
    "tpe_robustness_summary.json",
    "carrier_transfer_summary.json",
]


def prepare(tmp_path):
    data = tmp_path / "data"
    output = tmp_path / "campaign"
    manuscript = tmp_path / "manuscript"
    data.mkdir()
    output.mkdir()
    manuscript.mkdir()
    for name in PREREQUISITES:
        payload = "{}"
        if name == "carrier_transfer_summary.json":
            payload = """{
  "status": "complete",
  "test_evaluated": false,
  "selections": {
    "FD001": {
      "self_scaled_selected_cap": 90,
      "fixed_reference_selected_cap": 125,
      "profile": [
        {"cap": 90, "self_scaled_own_rmse": 7.0,
         "self_checkpoint_reference_rmse": 13.0,
         "fixed_reference_rmse": 12.0},
        {"cap": 125, "self_scaled_own_rmse": 10.0,
         "self_checkpoint_reference_rmse": 10.0,
         "fixed_reference_rmse": 9.0}
      ]
    },
    "FD002": {
      "self_scaled_selected_cap": 100,
      "fixed_reference_selected_cap": 115,
      "profile": [
        {"cap": 100, "self_scaled_own_rmse": 8.0,
         "self_checkpoint_reference_rmse": 12.0,
         "fixed_reference_rmse": 11.0},
        {"cap": 115, "self_scaled_own_rmse": 9.0,
         "self_checkpoint_reference_rmse": 10.0,
         "fixed_reference_rmse": 9.5}
      ]
    }
  }
}"""
        (output / name).write_text(payload, encoding="utf-8")
    return data, output, manuscript


def argv(data, output, manuscript):
    return [
        "wait_and_finalize_confirmatory.py",
        "--data_root",
        str(data),
        "--output_root",
        str(output),
        "--manuscript_dir",
        str(manuscript),
        "--poll_seconds",
        "1",
    ]


def command_labels(commands):
    labels = []
    for command, _ in commands:
        if command[0] == "latexmk":
            labels.append(f"latexmk:{command[-1]}")
        else:
            labels.append(Path(command[1]).name)
    return labels


def test_watcher_resumes_partial_official_state(tmp_path, monkeypatch):
    data, output, manuscript = prepare(tmp_path)
    (output / "immutable_test_plan.json").write_text("{}", encoding="utf-8")
    commands = []

    def record(command, cwd):
        commands.append((command, cwd))

    monkeypatch.setattr(watcher, "run_checked", record)
    monkeypatch.setattr(sys, "argv", argv(data, output, manuscript))
    assert watcher.main() == 0
    assert command_labels(commands) == [
        "create_selection_attempt_manifest.py",
        "run_qa_suite.py",
        "preflight_confirmatory.py",
        "run_confirmatory_final.py",
        "summarize_confirmatory.py",
        "generate_manuscript_assets.py",
        "build_cover_letter_docx.py",
        "build_highlights_docx.py",
        "latexmk:main.tex",
        "latexmk:supplementary_material.tex",
        "latexmk:title_page.tex",
        "latexmk:highlights.tex",
        "audit_submission.py",
        "write_submission_asset_manifest.py",
        "audit_submission.py",
    ]
    assert (manuscript / "FINAL_RESULTS_READY.tex").read_text(
        encoding="utf-8"
    ).startswith(
        "% Generated only after the locked official-result assets exist.\n"
        "\\finalresultstrue\n"
    )
    carrier_table = (
        manuscript / "generated_carrier_transfer_table.tex"
    ).read_text(encoding="utf-8")
    assert "Complete primary-split, single-seed cap profile" in carrier_table
    assert "FD001 & 90 & 7.00 & 13.00 & 12.00 & S" in carrier_table


def test_completed_test_is_summarized_without_relaunch(tmp_path, monkeypatch):
    data, output, manuscript = prepare(tmp_path)
    (output / "TEST_EVALUATED").write_text("{}", encoding="utf-8")
    (output / "final_test_summary.json").write_text("{}", encoding="utf-8")
    commands = []

    def record(command, cwd):
        commands.append((command, cwd))

    monkeypatch.setattr(watcher, "run_checked", record)
    monkeypatch.setattr(sys, "argv", argv(data, output, manuscript))
    assert watcher.main() == 0
    assert command_labels(commands) == [
        "create_selection_attempt_manifest.py",
        "run_qa_suite.py",
        "summarize_confirmatory.py",
        "generate_manuscript_assets.py",
        "build_cover_letter_docx.py",
        "build_highlights_docx.py",
        "latexmk:main.tex",
        "latexmk:supplementary_material.tex",
        "latexmk:title_page.tex",
        "latexmk:highlights.tex",
        "audit_submission.py",
        "write_submission_asset_manifest.py",
        "audit_submission.py",
    ]
    assert (manuscript / "FINAL_RESULTS_READY.tex").read_text(
        encoding="utf-8"
    ).startswith(
        "% Generated only after the locked official-result assets exist.\n"
        "\\finalresultstrue\n"
    )
    assert (
        manuscript / "generated_carrier_transfer_table.tex"
    ).is_file()


def test_terminal_carrier_failure_is_reported_without_selection_claim():
    summary = {
        "status": "terminal_failure",
        "selections": {
            "FD001": {
                "self_scaled_selected_cap": 90,
                "fixed_reference_selected_cap": 110,
                "profile": [
                    {
                        "cap": 90,
                        "self_scaled_own_rmse": 7.9,
                        "self_checkpoint_reference_rmse": 14.2,
                        "fixed_reference_rmse": 14.1,
                    },
                    {
                        "cap": 110,
                        "self_scaled_own_rmse": 10.3,
                        "self_checkpoint_reference_rmse": 10.7,
                        "fixed_reference_rmse": 10.7,
                    },
                ],
            }
        },
        "partial_profiles": {
            "FD001": [
                {
                    "cap": 90,
                    "self_scaled_own_rmse": 7.9,
                    "self_checkpoint_reference_rmse": 14.2,
                    "fixed_reference_rmse": 14.1,
                },
                {
                    "cap": 110,
                    "self_scaled_own_rmse": 10.3,
                    "self_checkpoint_reference_rmse": 10.7,
                    "fixed_reference_rmse": 10.7,
                },
            ],
            "FD002": [],
        },
        "missing_cells": [
            {"dataset": "FD002", "cap": 110},
            {"dataset": "FD002", "cap": 115},
        ],
    }

    table = watcher.carrier_transfer_table(summary)
    narrative = watcher.carrier_transfer_narrative(summary)

    assert "retry failed" in table
    assert "FD002 & 110 & -- & -- & -- & retry failed" in table
    assert "no selection is claimed" in narrative
    assert "FD002 cap 110, FD002 cap 115" in narrative
