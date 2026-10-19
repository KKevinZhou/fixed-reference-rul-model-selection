"""Validation-only, campaign-hashed TPE robustness study.

Sequence length is fixed so every trial uses the same deterministic validation
endpoints. TPE may vary the training cap, learning rate, batch size, and Huber
threshold. Separate studies adapt to the self-scaled or fixed-reference
objective under an equal trial budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import optuna
from optuna.trial import TrialState


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINER = SCRIPT_DIR / "confirmatory_rul_train.py"
MODEL = SCRIPT_DIR / "baselines" / "models_v3.py"
CAPS = [90, 100, 110, 115, 125]
HUBER = [0.5, 1.0, 1.5]
DEFAULT_SEQ = {"FD001": 50, "FD002": 50}


def stable_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["FD001", "FD002"])
    parser.add_argument(
        "--objective",
        required=True,
        choices=["self_scaled", "fixed_reference", "fixed_cap_125"],
    )
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--search_seed", type=int, default=20260724)
    parser.add_argument("--split_seed", type=int, default=20260724)
    parser.add_argument("--model_seed", type=int, default=42)
    parser.add_argument("--data_root", type=Path, default=Path(r"E:\NASA_Lab"))
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(r"E:\NASA_Lab\ress_confirmatory_v2"),
    )
    return parser.parse_args()


def build_manifest(args, screening):
    definition = {
        "protocol": "confirmatory-tpe-v2",
        "screening_campaign_id": screening["campaign_id"],
        "dataset": args.dataset,
        "objective": args.objective,
        "checkpoint_objective": (
            "self_scaled"
            if args.objective == "self_scaled"
            else "fixed_reference"
        ),
        "metric": (
            "own_rmse"
            if args.objective == "self_scaled"
            else "reference_rmse"
        ),
        "trials": args.trials,
        "search_seed": args.search_seed,
        "sampler": {
            "name": "Optuna TPESampler",
            "n_startup_trials": 6,
        },
        "split_seed": args.split_seed,
        "endpoint_seed": args.split_seed + 1000003,
        "model_seed": args.model_seed,
        "fixed_seq_len": DEFAULT_SEQ[args.dataset],
        "search_space": {
            "train_cap": (
                [125] if args.objective == "fixed_cap_125" else CAPS
            ),
            "lr": [2e-4, 1.2e-3, "log"],
            "batch": [64, 128],
            "huber_delta": HUBER,
        },
        "epochs": screening["epochs"],
        "patience": screening["patience"],
        "train_stride": screening["train_stride"],
        "validation_interval": screening["validation_interval"],
        "carrier": screening["carrier"],
        "tpe_script_sha256": sha256_file(Path(__file__).resolve()),
        "trainer_sha256": screening["trainer_sha256"],
        "model_sha256": screening["model_sha256"],
        "raw_train_sha256": screening["raw_train_sha256"][args.dataset],
        "test_evaluated": False,
    }
    definition["campaign_id"] = "tpev2-" + stable_hash(definition)[:16]
    return definition


def trial_params(trial, objective):
    return {
        "train_cap": trial.suggest_categorical(
            "train_cap",
            [125] if objective == "fixed_cap_125" else CAPS,
        ),
        "lr": trial.suggest_float("lr", 2e-4, 1.2e-3, log=True),
        "batch": trial.suggest_categorical("batch", [64, 128]),
        "huber_delta": trial.suggest_categorical("huber_delta", HUBER),
    }


def result_valid(path, manifest, params):
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    config = result.get("config", {})
    objective = manifest["checkpoint_objective"]
    metric = manifest["metric"]
    try:
        values = result["validation_summary"][objective][metric]["values"]
        finite = all(math.isfinite(float(value)) for value in values)
    except Exception:
        return False
    return (
        finite
        and result.get("campaign_id") == manifest["campaign_id"]
        and result.get("dataset") == manifest["dataset"]
        and result.get("validation_only") is True
        and result.get("test_evaluated") is False
        and result.get("split_seed") == manifest["split_seed"]
        and result.get("endpoint_seed") == manifest["endpoint_seed"]
        and result.get("model_seeds") == [manifest["model_seed"]]
        and int(result.get("train_cap")) == int(params["train_cap"])
        and int(config.get("seq_len")) == manifest["fixed_seq_len"]
        and math.isclose(float(config.get("lr")), float(params["lr"]))
        and int(config.get("batch")) == int(params["batch"])
        and math.isclose(
            float(config.get("huber_delta")),
            float(params["huber_delta"]),
        )
        and config.get("epochs") == manifest["epochs"]
        and config.get("patience") == manifest["patience"]
        and config.get("train_stride") == manifest["train_stride"]
        and config.get("validation_interval")
        == manifest["validation_interval"]
        and config.get("carrier") == manifest["carrier"]
        and result.get("source_fingerprints", {}).get("trainer_sha256")
        == manifest["trainer_sha256"]
        and result.get("source_fingerprints", {}).get("model_sha256")
        == manifest["model_sha256"]
        and result.get("data_fingerprints", {}).get("train_sha256")
        == manifest["raw_train_sha256"]
    )


def run_trial(args, manifest, trial_number, params):
    if (
        sha256_file(TRAINER) != manifest["trainer_sha256"]
        or sha256_file(MODEL) != manifest["model_sha256"]
    ):
        raise RuntimeError("Training source changed during TPE execution")
    param_hash = stable_hash(params)[:10]
    tag = (
        f"tpe_{args.objective}_q{args.search_seed}_"
        f"trial{trial_number:03d}_{param_hash}"
    )
    path = args.output_root / "results" / f"{args.dataset}_{tag}.json"
    if result_valid(path, manifest, params):
        return path
    if path.exists():
        raise RuntimeError(f"Invalid immutable TPE result: {path}")
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
        "--campaign_id",
        manifest["campaign_id"],
        "--tag",
        tag,
        "--train_cap",
        str(params["train_cap"]),
        "--reporting_cap",
        "125",
        "--split_seed",
        str(manifest["split_seed"]),
        "--endpoint_seed",
        str(manifest["endpoint_seed"]),
        "--seeds",
        "1",
        "--seed_base",
        str(manifest["model_seed"]),
        "--seq_len",
        str(manifest["fixed_seq_len"]),
        "--lr",
        str(params["lr"]),
        "--batch",
        str(params["batch"]),
        "--huber_delta",
        str(params["huber_delta"]),
        "--epochs",
        str(manifest["epochs"]),
        "--patience",
        str(manifest["patience"]),
        "--train_stride",
        str(manifest["train_stride"]),
        "--validation_interval",
        str(manifest["validation_interval"]),
        "--carrier",
        manifest["carrier"],
        "--validation_only",
    ]
    log = (
        args.output_root
        / "tpe_logs"
        / f"{args.dataset}_{tag}_{manifest['campaign_id']}.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"RUN {args.dataset} {args.objective} trial={trial_number} "
        f"params={params}",
        flush=True,
    )
    with log.open("x", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=SCRIPT_DIR,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"TPE trial failed; see {log}")
    if not result_valid(path, manifest, params):
        raise RuntimeError(f"TPE result failed validation: {path}")
    return path


def main():
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    screening_path = args.output_root / "campaign_manifest.json"
    if not screening_path.exists():
        raise RuntimeError("Screening campaign manifest is missing")
    screening = json.loads(screening_path.read_text(encoding="utf-8"))
    if (
        screening.get("trainer_sha256") != sha256_file(TRAINER)
        or screening.get("model_sha256") != sha256_file(MODEL)
    ):
        raise RuntimeError("Trainer or model source changed after screening")
    manifest = build_manifest(args, screening)
    manifest_path = (
        args.output_root
        / "tpe_manifests"
        / f"{args.dataset}_{args.objective}.json"
    )
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("Existing TPE manifest differs")
    else:
        exclusive_json(manifest_path, manifest)

    study_name = manifest["campaign_id"]
    database = (
        args.output_root
        / "optuna"
        / f"{args.dataset}_{args.objective}_{study_name}.sqlite3"
    )
    database.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(
        seed=args.search_seed,
        n_startup_trials=6,
    )
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        storage=f"sqlite:///{database.as_posix()}",
        load_if_exists=True,
    )

    def objective(trial):
        params = trial_params(trial, args.objective)
        path = run_trial(args, manifest, trial.number, params)
        result = json.loads(path.read_text(encoding="utf-8"))
        value = float(
            result["validation_summary"][manifest["checkpoint_objective"]][
                manifest["metric"]
            ]["mean"]
        )
        trial.set_user_attr("result_path", str(path))
        trial.set_user_attr("result_sha256", sha256_file(path))
        trial.set_user_attr(
            "reference_rmse",
            float(
                result["validation_summary"]["fixed_reference"][
                    "reference_rmse"
                ]["mean"]
            ),
        )
        trial.set_user_attr(
            "own_rmse",
            float(
                result["validation_summary"]["self_scaled"]["own_rmse"][
                    "mean"
                ]
            ),
        )
        trial.set_user_attr(
            "elapsed_seconds",
            float(result["elapsed_seconds"]),
        )
        return value

    existing_trials = list(study.trials)
    if len(existing_trials) > args.trials:
        raise RuntimeError(
            f"Study already contains {len(existing_trials)} trials; "
            f"the immutable budget is exactly {args.trials}"
        )
    invalid_states = [
        trial
        for trial in existing_trials
        if trial.state != TrialState.COMPLETE
    ]
    if invalid_states:
        raise RuntimeError(
            "TPE database contains non-complete trials; audit rather than "
            "silently replacing budgeted trials: "
            + ", ".join(
                f"{trial.number}:{trial.state.name}"
                for trial in invalid_states
            )
        )
    remaining = args.trials - len(existing_trials)
    if remaining:
        study.optimize(objective, n_trials=remaining)
    all_trials = list(study.trials)
    complete = [
        trial for trial in all_trials
        if trial.state == TrialState.COMPLETE
    ]
    if (
        len(all_trials) != args.trials
        or len(complete) != args.trials
    ):
        raise RuntimeError(
            f"Study has {len(all_trials)} total and {len(complete)} "
            f"complete trials; expected exactly {args.trials} of each"
        )
    best = min(complete, key=lambda trial: trial.value)
    summary = {
        "protocol": "confirmatory-tpe-lock-v2",
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": stable_hash(manifest),
        "dataset": args.dataset,
        "objective": args.objective,
        "metric": manifest["metric"],
        "complete_trials": len(complete),
        "failed_trials": sum(
            trial.state == TrialState.FAIL for trial in study.trials
        ),
        "running_trials": sum(
            trial.state == TrialState.RUNNING for trial in study.trials
        ),
        "waiting_trials": sum(
            trial.state == TrialState.WAITING for trial in study.trials
        ),
        "sampler": {
            "name": "Optuna TPESampler",
            "n_startup_trials": 6,
            "seed": args.search_seed,
        },
        "budget_definition": (
            "equal number of complete trials; wall-clock/GPU time is "
            "recorded separately and is not claimed equal"
        ),
        "total_elapsed_seconds": float(
            sum(
                trial.user_attrs["elapsed_seconds"]
                for trial in complete
            )
        ),
        "best_trial": best.number,
        "best_value": float(best.value),
        "best_params": best.params,
        "best_result_path": best.user_attrs["result_path"],
        "test_evaluated": False,
        "trial_table": [
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": (
                    float(trial.value)
                    if trial.value is not None
                    else None
                ),
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
            for trial in study.trials
        ],
    }
    summary_path = (
        args.output_root
        / "tpe_locks"
        / f"{args.dataset}_{args.objective}.json"
    )
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing != summary:
            raise RuntimeError("Existing immutable TPE lock differs")
    else:
        exclusive_json(summary_path, summary)
    atomic_json(
        args.output_root
        / "tpe_status"
        / f"{args.dataset}_{args.objective}.json",
        summary,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
