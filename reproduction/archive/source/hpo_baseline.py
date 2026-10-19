#!/usr/bin/env python
"""
Matched-compute HPO baseline harness for the RUL-cap study.

The script compares random search and TPE with the legacy signal-feature
heuristic using the same C-MAPSS trainer, split, seed count, and compute budget.
Legacy ``PHR`` identifiers are retained in output fields solely for backward
compatibility with the archived experiment files.

It searches the same five hyperparameters PHR routes:
    SEQ_LEN, MAX_RUL, LR, huber_delta, ema_span.

OUTPUT
------
  hpo_runs/<dataset>_<method>_trials.csv   one row per trial (config + test RMSE)
  hpo_runs/<dataset>_<method>_summary.json best-so-far curve + trials-to-PHR

USAGE (run on the GPU box — NOT runnable on the CPU-only checkout)
-----
  # 20-trial random search on FD001, 2 seeds/trial (fast, for the curve):
  python hpo_baseline.py --dataset FD001 --method random --trials 20 --seeds 2

  # TPE (needs: pip install optuna):
  python hpo_baseline.py --dataset FD001 --method tpe --trials 20 --seeds 2

  # Match the paper's 5-seed protocol for the final comparison rows:
  python hpo_baseline.py --dataset FD002 --method random --trials 25 --seeds 5

NOTE: each trial trains `--seeds` models via train_v3_regression.py, so cost =
trials x seeds x (~12 min on the paper's GPU). Use 1-2 seeds for the search
curve; re-run the single best config at 5 seeds for the reported number.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# Windows consoles default to cp1252; the trainer prints non-ASCII (arrows), which
# crashes the child process on encode. Force UTF-8 stdio in every subprocess.
_ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts_v3"
OUT = ROOT / "hpo_runs"
OUT.mkdir(exist_ok=True)

# PHR-1D reference RMSE (paper Table tab:main) — the bar the HPO baseline must reach.
PHR_1D_RMSE = {"FD001": 13.01, "FD002": 10.57, "FD003": 10.63, "FD004": 12.86}

# Search space = the same five HPs PHR routes, over the ranges the paper swept.
SPACE = {
    "seq_len":     ("int",     20,   90),
    "max_rul":     ("int",    110,  135),
    "lr":          ("logfloat", 3e-4, 1.5e-3),
    "huber_delta": ("float",   0.5,  2.0),
    "ema_span":    ("int",      1,    5),
}

# PHR routed (theoretical) config per subset = the physics-feature POINT estimate.
# In feature-informed mode the search is CENTERED on this (a prior/band), instead of
# blind full-range search. Tests whether the features make the search more efficient
# even though the point estimate itself is suboptimal.
ROUTED = {
    "FD001": {"seq_len": 30, "max_rul": 125, "lr": 5e-4},
    "FD002": {"seq_len": 78, "max_rul": 115, "lr": 1e-3},
    "FD003": {"seq_len": 34, "max_rul": 120, "lr": 1e-3},
    "FD004": {"seq_len": 86, "max_rul": 115, "lr": 5e-4},
}


def sample_route_adjust(rng, sub):
    """User's hybrid: random init, then 'adjust with the physics rule' -- blend the two
    structural HPs halfway toward the routed value. Tests whether nudging a free random
    config toward the (biased) rule helps or hurts vs pure random."""
    base = sample_random(rng, {})
    r = ROUTED[sub]
    base["seq_len"] = int(np.clip(round(0.5 * base["seq_len"] + 0.5 * r["seq_len"]), 15, 100))
    base["max_rul"] = int(np.clip(round(0.5 * base["max_rul"] + 0.5 * r["max_rul"]), 100, 140))
    return base


def sample_feature_informed(rng, sub):
    """Search CENTERED on the physics-feature prediction: seq/max/lr in a band around
    the routed value (feature prior), huber/ema left uninformed (not feature-derived)."""
    r = ROUTED[sub]
    seq = int(np.clip(round(rng.uniform(0.6, 1.5) * r["seq_len"]), 15, 100))
    mx  = int(np.clip(round(rng.uniform(0.90, 1.15) * r["max_rul"]), 100, 140))
    lr  = float(np.clip(np.exp(rng.uniform(np.log(0.5 * r["lr"]), np.log(2.0 * r["lr"]))),
                        3e-4, 1.5e-3))
    return {"seq_len": seq, "max_rul": mx, "lr": lr,
            "huber_delta": float(rng.uniform(0.5, 2.0)),
            "ema_span": int(rng.integers(1, 6))}


def sample_random(rng, fixed=None):
    fixed = fixed or {}
    cfg = {}
    for name, (kind, lo, hi) in SPACE.items():
        if name in fixed:
            cfg[name] = fixed[name]
            continue
        if kind == "int":
            cfg[name] = int(rng.integers(lo, hi + 1))
        elif kind == "logfloat":
            cfg[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            cfg[name] = float(rng.uniform(lo, hi))
    return cfg


def run_trial(dataset, cfg, seeds, trial_id, tag_prefix="hpo"):
    """Train one config via the existing trainer; return its test ensemble RMSE."""
    tag = f"{tag_prefix}_{trial_id}"
    cmd = [
        sys.executable, str(ROOT / "train_v3_regression.py"),
        "--dataset", dataset, "--variant", "V_A", "--tag", tag,
        "--seeds", str(seeds), "--use_swa",
        "--seq_len", str(cfg["seq_len"]),
        "--max_rul", str(cfg["max_rul"]),
        "--lr", f"{cfg['lr']:.6g}",
        "--huber_delta", f"{cfg['huber_delta']:.4g}",
        "--ema_span", str(cfg["ema_span"]),
    ]
    print(f"\n[trial {trial_id}] {cfg}")
    subprocess.run(cmd, check=True, env=_ENV, cwd=str(ROOT))
    art = ART / f"{dataset}_V_A_{tag}.json"
    with open(art) as f:
        res = json.load(f)
    return float(res["ensemble_rmse"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(PHR_1D_RMSE.keys()))
    ap.add_argument("--method", default="random", choices=["random", "tpe"])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=2, help="seeds per trial (use 5 for final rows)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the search itself")
    ap.add_argument("--fix_max_rul", type=float, default=None,
                    help="Pin MAX_RUL to PHR's routed value and search the rest — "
                         "controls for the label-censoring artefact so the comparison is fair.")
    ap.add_argument("--feature_informed", action="store_true",
                    help="Center the search on the physics-feature routed config (band prior) "
                         "instead of blind full-range random. Tests the 'system' value of the features.")
    ap.add_argument("--route_adjust", action="store_true",
                    help="Random init then blend structural HPs halfway toward the routed value "
                         "(the 'random-first, rule-adjust' hybrid).")
    ap.add_argument("--max_rul_min", type=int, default=110,
                    help="Low end of the MAX_RUL search range (default 110). Set to 90 to let the "
                         "tuner drift into the censoring regime and demonstrate cap-gaming end-to-end.")
    ap.add_argument("--max_rul_max", type=int, default=135,
                    help="High end of the MAX_RUL search range (default 135).")
    args = ap.parse_args()

    # Widen/narrow the MAX_RUL search band if requested (default preserves prior behaviour).
    SPACE["max_rul"] = ("int", args.max_rul_min, args.max_rul_max)

    target = PHR_1D_RMSE[args.dataset]
    rows, best, best_cfg, trials_to_target = [], float("inf"), None, None
    fixed = {"max_rul": int(args.fix_max_rul)} if args.fix_max_rul is not None else {}
    capfree = (args.max_rul_min, args.max_rul_max) != (110, 135) and not fixed
    label = args.method + (f"_fixmax{int(args.fix_max_rul)}" if fixed else "") + ("_fi" if args.feature_informed else "") + ("_radj" if args.route_adjust else "") + (f"_capfree{args.max_rul_min}" if capfree else "")
    tag_prefix = f"hpo_{label}"

    if args.method == "tpe":
        try:
            import optuna
        except ImportError:
            sys.exit("optuna not installed: pip install optuna  (or use --method random)")
        study = optuna.create_study(direction="minimize",
                                     sampler=optuna.samplers.TPESampler(seed=args.seed))

        def objective(trial):
            cfg = {
                "seq_len":     trial.suggest_int("seq_len", 20, 90),
                "max_rul":     fixed.get("max_rul") or trial.suggest_int("max_rul", args.max_rul_min, args.max_rul_max),
                "lr":          trial.suggest_float("lr", 3e-4, 1.5e-3, log=True),
                "huber_delta": trial.suggest_float("huber_delta", 0.5, 2.0),
                "ema_span":    trial.suggest_int("ema_span", 1, 5),
            }
            rmse = run_trial(args.dataset, cfg, args.seeds, trial.number, tag_prefix)
            rows.append({"trial": trial.number, **cfg, "rmse": rmse})
            return rmse

        study.optimize(objective, n_trials=args.trials)
    else:
        rng = np.random.default_rng(args.seed)
        for i in range(args.trials):
            if args.feature_informed:
                cfg = sample_feature_informed(rng, args.dataset)
            elif args.route_adjust:
                cfg = sample_route_adjust(rng, args.dataset)
            else:
                cfg = sample_random(rng, fixed)
            rmse = run_trial(args.dataset, cfg, args.seeds, i, tag_prefix)
            rows.append({"trial": i, **cfg, "rmse": rmse})

    # best-so-far curve + trials-to-reach-PHR
    best_curve = []
    running = float("inf")
    for i, r in enumerate(rows):
        if r["rmse"] < running:
            running = r["rmse"]; best_cfg = {k: r[k] for k in SPACE}
        best_curve.append(running)
        if trials_to_target is None and running <= target:
            trials_to_target = i + 1
        best = running

    csv_path = OUT / f"{args.dataset}_{label}_trials.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial", *SPACE, "rmse"])
        w.writeheader(); w.writerows(rows)

    summary = {
        "dataset": args.dataset, "method": args.method, "label": label,
        "fixed": fixed, "trials": args.trials, "seeds_per_trial": args.seeds,
        "phr_1d_rmse": target,
        "best_rmse": best, "best_cfg": best_cfg,
        "trials_to_reach_phr_1d": trials_to_target,
        "best_so_far_curve": best_curve,
    }
    with open(OUT / f"{args.dataset}_{label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"{args.method} on {args.dataset}: best RMSE {best:.3f} (PHR-1D {target})")
    if trials_to_target:
        print(f"Reached PHR-1D RMSE after {trials_to_target} trials "
              f"(= {trials_to_target * args.seeds} single-seed trainings).")
        print(f"PHR-1D uses 5 trials => honest speedup ~{trials_to_target / 5:.1f}x over {args.method} search.")
    else:
        print(f"Did NOT reach PHR-1D RMSE in {args.trials} trials "
              f"=> PHR-1D is at least as good as {args.trials}-trial {args.method} search.")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
