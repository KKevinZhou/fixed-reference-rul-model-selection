"""Validation-only TPE for comparing self-scaled and fixed-reference objectives.

The trial objective is read exclusively from ``strict_rul_train.py`` results
created with ``--validation_only``.  The test split is evaluated once, and only
if ``--final_test`` is explicitly supplied after the study has selected a
configuration.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import optuna


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINER = SCRIPT_DIR / "strict_rul_train.py"
CAPS = [90, 100, 110, 115, 120, 125]
SEQUENCES = [30, 40, 50, 60, 70, 80]
HUBER_DELTAS = [0.5, 0.75, 1.0, 1.25, 1.5]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument(
        "--objective",
        required=True,
        choices=["own_rmse", "reference_rmse"],
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--trial_seeds", type=int, default=1)
    parser.add_argument("--final_seeds", type=int, default=5)
    parser.add_argument("--search_seed", type=int, default=20260724)
    parser.add_argument("--split_seed", type=int, default=20260724)
    parser.add_argument("--model_seed_base", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--data_root", type=Path, default=Path(r"E:\NASA_Lab"))
    parser.add_argument(
        "--output_root", type=Path, default=Path(r"E:\NASA_Lab\ress_strict")
    )
    parser.add_argument("--final_test", action="store_true")
    return parser.parse_args()


def run_training(args, tag, params, validation_only, seeds):
    command = [
        sys.executable,
        "-u",
        str(TRAINER),
        "--dataset",
        args.dataset,
        "--data_root",
        str(args.data_root),
        "--output_root",
        str(args.output_root),
        "--tag",
        tag,
        "--train_cap",
        str(params["train_cap"]),
        "--reporting_cap",
        "125",
        "--checkpoint_metric",
        args.objective,
        "--seq_len",
        str(params["seq_len"]),
        "--lr",
        str(params["lr"]),
        "--batch",
        str(params["batch"]),
        "--huber_delta",
        str(params["huber_delta"]),
        "--seeds",
        str(seeds),
        "--seed_base",
        str(args.model_seed_base),
        "--split_seed",
        str(args.split_seed),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--disable_het",
        "--disable_length",
    ]
    if validation_only:
        command.append("--validation_only")

    log_dir = args.output_root / "hpo_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.dataset}_{tag}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=args.data_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    result_path = args.output_root / "results" / f"{args.dataset}_{tag}.json"
    if process.returncode != 0 or not result_path.exists():
        raise RuntimeError(f"Training failed; see {log_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    short_objective = "own" if args.objective == "own_rmse" else "ref125"
    study_name = (
        f"{args.dataset}_strict_{short_objective}_"
        f"s{args.split_seed}_q{args.search_seed}"
    )
    storage_path = args.output_root / "optuna" / f"{study_name}.sqlite3"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_path.as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=args.search_seed)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial):
        params = {
            "train_cap": trial.suggest_categorical("train_cap", CAPS),
            "seq_len": trial.suggest_categorical("seq_len", SEQUENCES),
            "lr": trial.suggest_float("lr", 2e-4, 1.2e-3, log=True),
            "batch": trial.suggest_categorical("batch", [64, 128]),
            "huber_delta": trial.suggest_categorical(
                "huber_delta", HUBER_DELTAS
            ),
        }
        tag = f"hpo_{short_objective}_q{args.search_seed}_trial{trial.number:03d}"
        result_path = args.output_root / "results" / f"{args.dataset}_{tag}.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = run_training(
                args,
                tag,
                params,
                validation_only=True,
                seeds=args.trial_seeds,
            )
        if not result.get("validation_only", False):
            raise RuntimeError("HPO trial unexpectedly contains a test evaluation")
        score = result["validation_summary"][args.objective]["mean"]
        trial.set_user_attr("result_path", str(result_path))
        trial.set_user_attr(
            "reference_rmse",
            result["validation_summary"]["reference_rmse"]["mean"],
        )
        trial.set_user_attr(
            "own_rmse", result["validation_summary"]["own_rmse"]["mean"]
        )
        trial.set_user_attr(
            "original_rmse",
            result["validation_summary"]["original_rmse"]["mean"],
        )
        return score

    remaining = max(0, args.trials - len(study.trials))
    if remaining:
        study.optimize(objective, n_trials=remaining)

    best = study.best_trial
    summary = {
        "protocol": "strict-rul-cap-hpo-v1",
        "study_name": study_name,
        "dataset": args.dataset,
        "objective": args.objective,
        "reporting_cap": 125,
        "search_space": {
            "train_cap": CAPS,
            "seq_len": SEQUENCES,
            "lr": [2e-4, 1.2e-3, "log"],
            "batch": [64, 128],
            "huber_delta": HUBER_DELTAS,
        },
        "split_seed": args.split_seed,
        "search_seed": args.search_seed,
        "trial_seeds": args.trial_seeds,
        "n_trials": len(study.trials),
        "best_value": best.value,
        "best_params": best.params,
        "best_trial": best.number,
        "test_evaluated_during_search": False,
    }

    if args.final_test:
        final_tag = f"hpo_{short_objective}_q{args.search_seed}_final"
        final_result = run_training(
            args,
            final_tag,
            best.params,
            validation_only=False,
            seeds=args.final_seeds,
        )
        summary["final_test_result"] = str(
            args.output_root
            / "results"
            / f"{args.dataset}_{final_tag}.json"
        )
        summary["final_test_summary"] = final_result["test_summary"]
        summary["final_ensemble_test"] = final_result["ensemble_test"]

    summary_path = args.output_root / f"{study_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
