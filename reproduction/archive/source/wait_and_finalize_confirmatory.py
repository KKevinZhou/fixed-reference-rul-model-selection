"""Wait for validation locks, then run the single fail-closed final pipeline.

This watcher never publishes or uploads anything.  It starts the official-test
wrapper only after every validation/HPO completion artifact exists and a strict
preflight succeeds.  Any partial official-test state is left for manual review.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent


def stamp(message: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    print(f"[{now}] {message}", flush=True)


def run_checked(command: list[str], cwd: Path) -> None:
    stamp("RUN " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def carrier_transfer_table(summary: dict) -> str:
    terminal_failure = summary.get("status") == "terminal_failure"
    selections = summary.get("selections", {})
    profiles = (
        summary.get("partial_profiles", {})
        if terminal_failure
        else {
            dataset: value.get("profile", [])
            for dataset, value in selections.items()
        }
    )
    caption = (
        "Plan-bound primary-split, single-seed cap profile for the "
        "validation-only second carrier. Rows marked retry failed retained "
        "a non-finite loss after one exact same-configuration sequential "
        "retry; no selection is claimed for an incomplete subset."
        if terminal_failure
        else (
            "Complete primary-split, single-seed cap profile for the "
            "validation-only second carrier. Self-own uses the candidate "
            "target and output clip; Self-ref rescales the self-selected "
            "checkpoint to the common reporting target and output clip; Fixed-ref is the "
            "fixed-reference-selected checkpoint. S and R mark the caps "
            "selected by the self-scaled and fixed-reference objectives, "
            "respectively."
        )
    )
    lines = [
        "% Generated from carrier_transfer_summary.json; do not edit.",
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{sup:tab:carrier-transfer}",
        "\\small",
        "\\begin{tabular}{@{}lrrrrl@{}}",
        "\\toprule",
        "Subset & Cap & Self-own RMSE & Self-ref RMSE & Fixed-ref RMSE & "
        "Selected \\\\",
        "\\midrule",
    ]
    for dataset in ("FD001", "FD002"):
        selection = selections.get(dataset)
        self_cap = (
            int(selection["self_scaled_selected_cap"])
            if selection
            else None
        )
        fixed_cap = (
            int(selection["fixed_reference_selected_cap"])
            if selection
            else None
        )
        profile = {
            int(row["cap"]): row
            for row in profiles.get(dataset, [])
        }
        row_caps = (
            (90, 100, 110, 115, 125)
            if terminal_failure
            else tuple(sorted(profile))
        )
        for cap in row_caps:
            row = profile.get(cap)
            if row is None:
                lines.append(
                    f"{dataset} & {cap} & -- & -- & -- & retry failed \\\\"
                )
                continue
            selected = []
            if cap == self_cap:
                selected.append("S")
            if cap == fixed_cap:
                selected.append("R")
            lines.append(
                f"{dataset} & {cap} & "
                f"{float(row['self_scaled_own_rmse']):.2f} & "
                f"{float(row['self_checkpoint_reference_rmse']):.2f} & "
                f"{float(row['fixed_reference_rmse']):.2f} & "
                f"{'/'.join(selected) if selected else '--'} \\\\"
            )
        if dataset == "FD001":
            lines.append("\\addlinespace")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def carrier_transfer_narrative(summary: dict) -> str:
    selections = summary.get("selections", {})
    if summary.get("status") == "complete":
        fd1 = selections["FD001"]
        fd2 = selections["FD002"]
        return (
            "On the validation-only dual-stream carrier, self-scaled versus "
            "fixed-reference selection chose caps "
            f"{int(fd1['self_scaled_selected_cap'])} versus "
            f"{int(fd1['fixed_reference_selected_cap'])} on FD001 and "
            f"{int(fd2['self_scaled_selected_cap'])} versus "
            f"{int(fd2['fixed_reference_selected_cap'])} on FD002. "
            "These cells are a descriptive architecture-sensitivity check "
            "and do not alter any locked standard-test decision."
        )
    complete = []
    for dataset, value in sorted(selections.items()):
        complete.append(
            f"{dataset} selected "
            f"{int(value['self_scaled_selected_cap'])} versus "
            f"{int(value['fixed_reference_selected_cap'])}"
        )
    failed = ", ".join(
        f"{row['dataset']} cap {int(row['cap'])}"
        for row in summary.get("missing_cells", [])
    )
    complete_text = "; ".join(complete) or "no subset was complete"
    return (
        "The validation-only second-carrier check remained descriptive: "
        f"{complete_text}. For {failed}, a non-finite training loss recurred "
        "after the single pre-specified exact-configuration sequential "
        "retry, so no selection is claimed for the incomplete subset and "
        "no standard-test decision was changed."
    )


def process_alive(pid: int) -> bool:
    if not isinstance(pid, int) or pid < 1:
        return False
    if os.name == "nt":
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(
                handle, ctypes.byref(exit_code)
            ):
                return False
            return exit_code.value == 259
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--manuscript_dir", type=Path, required=True)
    parser.add_argument("--poll_seconds", type=int, default=30)
    parser.add_argument("--final_seeds", type=int, default=5)
    args = parser.parse_args()

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "FINALIZATION_WATCHER_LOCK.json"
    lock_payload = {
        "protocol": "confirmatory-finalization-watcher-v1",
        "pid": os.getpid(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
    }
    if lock_path.exists():
        try:
            previous = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"Unreadable finalization watcher lock: {lock_path}"
            ) from exc
        previous_pid = previous.get("pid")
        if process_alive(previous_pid):
            raise RuntimeError(
                f"Finalization watcher is already active at PID {previous_pid}"
            )
        stale_path = lock_path.with_name(
            f"{lock_path.stem}.stale-{previous_pid}-{int(time.time())}.json"
        )
        os.replace(lock_path, stale_path)
        stamp(f"RECOVERED stale watcher lock as {stale_path.name}")
    with lock_path.open("x", encoding="utf-8") as handle:
        json.dump(lock_payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    prerequisites = [
        root / "cap_sweep_summary.json",
        root / "stability_summary.json",
        root / "final_cap_decisions.json",
        root / "TPE_SUITE_COMPLETE",
        root / "tpe_robustness_summary.json",
        root / "carrier_transfer_summary.json",
    ]
    stamp("WAIT validation/HPO prerequisites")
    while True:
        missing = [path.name for path in prerequisites if not path.is_file()]
        if not missing:
            break
        stamp("WAIT missing=" + ",".join(missing))
        time.sleep(args.poll_seconds)

    run_checked(
        [
            sys.executable,
            str(HERE / "create_selection_attempt_manifest.py"),
            "--output-root",
            str(root),
        ],
        HERE,
    )
    run_checked(
        [
            sys.executable,
            str(HERE / "run_qa_suite.py"),
            "--output-root",
            str(root),
        ],
        HERE,
    )

    evaluated = root / "TEST_EVALUATED"
    plan = root / "immutable_test_plan.json"
    started = root / "TEST_EVALUATION_STARTED.json"
    final_summary = root / "final_test_summary.json"

    if evaluated.is_file():
        if not final_summary.is_file():
            raise RuntimeError("TEST_EVALUATED exists without final_test_summary.json")
        stamp("Official test already complete; proceeding to strict summaries")
    else:
        partial = [path for path in (plan, started, final_summary) if path.exists()]
        if partial:
            stamp(
                "RESUME strict immutable official-test state="
                + ",".join(path.name for path in partial)
            )
        run_checked(
            [
                sys.executable,
                str(HERE / "preflight_confirmatory.py"),
                "--data_root",
                str(args.data_root.resolve()),
                "--output_root",
                str(root),
            ],
            HERE,
        )
        run_checked(
            [
                sys.executable,
                str(HERE / "run_confirmatory_final.py"),
                "--data_root",
                str(args.data_root.resolve()),
                "--output_root",
                str(root),
                "--final_seeds",
                str(args.final_seeds),
            ],
            HERE,
        )

    run_checked(
        [
            sys.executable,
            str(HERE / "summarize_confirmatory.py"),
            "--data_root",
            str(args.data_root.resolve()),
            "--output_root",
            str(root),
        ],
        HERE,
    )
    run_checked(
        [
            sys.executable,
            str(HERE / "generate_manuscript_assets.py"),
            "--summary_dir",
            str(root / "summaries"),
            "--manuscript_dir",
            str(args.manuscript_dir.resolve()),
        ],
        HERE,
    )
    run_checked(
        [
            sys.executable,
            str(
                args.manuscript_dir.resolve()
                / "build_cover_letter_docx.py"
            ),
        ],
        args.manuscript_dir.resolve(),
    )
    run_checked(
        [
            sys.executable,
            str(
                args.manuscript_dir.resolve()
                / "build_highlights_docx.py"
            ),
        ],
        args.manuscript_dir.resolve(),
    )
    final_results_marker = (
        args.manuscript_dir.resolve() / "FINAL_RESULTS_READY.tex"
    )
    carrier_transfer = json.loads(
        (root / "carrier_transfer_summary.json").read_text(encoding="utf-8")
    )
    if (
        carrier_transfer.get("status")
        not in {"complete", "terminal_failure"}
        or carrier_transfer.get("test_evaluated") is not False
    ):
        raise RuntimeError("Carrier-transfer summary is incomplete or invalid")
    transfer_selections = carrier_transfer.get("selections", {})
    if (
        carrier_transfer.get("status") == "complete"
        and set(transfer_selections) != {"FD001", "FD002"}
    ):
        raise RuntimeError("Carrier-transfer dataset set is invalid")
    atomic_text(
        args.manuscript_dir.resolve()
        / "generated_carrier_transfer_table.tex",
        carrier_transfer_table(carrier_transfer),
    )
    narrative = carrier_transfer_narrative(carrier_transfer)
    marker_text = (
        "% Generated only after the locked official-result assets exist.\n"
        "\\finalresultstrue\n"
        f"\\def\\CarrierTransferNarrative{{{narrative}}}\n"
    )
    if final_results_marker.exists():
        if final_results_marker.read_text(encoding="utf-8") != marker_text:
            raise RuntimeError(
                "Existing FINAL_RESULTS_READY.tex has unexpected content"
            )
    else:
        temporary_marker = final_results_marker.with_suffix(".tex.tmp")
        temporary_marker.write_text(marker_text, encoding="utf-8")
        temporary_marker.replace(final_results_marker)
    for stem in (
        "main.tex",
        "supplementary_material.tex",
        "title_page.tex",
        "highlights.tex",
    ):
        run_checked(
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                stem,
            ],
            args.manuscript_dir.resolve(),
        )
    run_checked(
        [
            sys.executable,
            str(args.manuscript_dir.resolve() / "audit_submission.py"),
            "--campaign-root",
            str(root),
        ],
        args.manuscript_dir.resolve(),
    )
    run_checked(
        [
            sys.executable,
            str(
                args.manuscript_dir.resolve()
                / "write_submission_asset_manifest.py"
            ),
        ],
        args.manuscript_dir.resolve(),
    )
    run_checked(
        [
            sys.executable,
            str(args.manuscript_dir.resolve() / "audit_submission.py"),
            "--campaign-root",
            str(root),
        ],
        args.manuscript_dir.resolve(),
    )
    stamp("RESULTS_FINALIZATION_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
