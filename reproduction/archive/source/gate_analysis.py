#!/usr/bin/env python
"""
Saturation and maintenance-cost sensitivity analysis for the RUL-cap study.
For each candidate cap, compare cap-dependent reported RMSE with RMSE against
the common uncapped target and evaluate an asymmetric maintenance-cost proxy.

Maintenance cost per unit (predict time-to-failure = y_hat, true = y):
  y_hat <= y  -> maintain BEFORE failure: waste (y - y_hat) of useful life   cost = 1*(y-y_hat)
  y_hat >  y  -> maintain AFTER failure : unplanned failure                    cost = rho*(y_hat-y)
Sweep rho = c_fail/c_waste. (Also report NASA-score against uncapped truth.)

"""
import glob, json, os, re
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DS = os.environ.get("GATE_DS", "FD001")
CONV = 125.0  # canonical C-MAPSS R_early

def nasa_score(pred, true):
    d = np.asarray(pred) - np.asarray(true)
    return float(np.sum(np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1)))

def maint_cost(pred, true, rho):
    d = np.asarray(pred) - np.asarray(true)  # >0 late (failure), <0 early (waste)
    return float(np.sum(np.where(d <= 0, -d, rho * d)))

rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "results", f"{DS}_V_A_gate_cap*.json")),
                key=lambda p: int(re.search(r"gate_cap(\d+)", p).group(1))):
    d = json.load(open(f))
    cap = d.get("train_cap") or float(re.search(r"gate_cap(\d+)", f).group(1))
    yhat = np.array(d["ensemble_unit_preds"], float)
    y = np.array(d["test_true_uncapped"], float)
    rows.append(dict(
        cap=cap, yhat=yhat, y=y,
        rmse_uncapped=float(np.sqrt(np.mean((yhat - y) ** 2))),          # score vs fixed uncapped ref
        rmse_capped=float(np.sqrt(np.mean((np.clip(yhat,0,cap)-np.clip(y,0,cap))**2))),  # cap-dependent (gameable)
        pct_saturated=float(np.mean(y > cap) * 100),
    ))

if not rows:
    raise SystemExit(f"no gate_cap artifacts for {DS} yet")

caps = [r["cap"] for r in rows]
print(f"=== {DS} GATE: cap sweep (n_units={len(rows[0]['y'])}, max true RUL={rows[0]['y'].max():.0f}) ===\n")
print(f"{'cap':>5} {'%saturated':>11} {'RMSE(reported)':>16} {'RMSE(uncapped)':>16}")
for r in rows:
    print(f"{r['cap']:>5.0f} {r['pct_saturated']:>11.0f} {r['rmse_capped']:>16.2f} {r['rmse_uncapped']:>16.2f}")

rmse_opt_cap = min(rows, key=lambda r: r["rmse_capped"])["cap"]  # what a naive RMSE search picks
print(f"\nRMSE-optimal cap (naive, capped-RMSE argmin): {rmse_opt_cap:.0f}   |   convention: {CONV:.0f}")

print(f"\n=== cost-optimal cap vs rho (rho = failure-cost / wasted-life-cost) ===")
print(f"{'rho':>5} {'cost-opt cap':>13} {'cost@opt':>10} {'cost@125':>10} {'cost@RMSEopt':>13}  {'sep from 125?':>13}")
kill_votes = []
for rho in [1, 2, 3, 5, 10, 20, 50]:
    costs = {r["cap"]: maint_cost(r["yhat"], r["y"], rho) for r in rows}
    opt = min(costs, key=costs.get)
    c125 = costs.get(CONV, float("nan"))
    cR = costs[rmse_opt_cap]
    sep = "YES" if abs(opt - CONV) > 1e-6 else "no(=125)"
    win_over_rmseopt = (cR - costs[opt]) / cR * 100 if cR else 0
    print(f"{rho:>5} {opt:>13.0f} {costs[opt]:>10.0f} {c125:>10.0f} {cR:>13.0f}  {sep:>13}  (win vs RMSEopt {win_over_rmseopt:+.0f}%)")
    kill_votes.append(abs(opt - CONV) < 1e-6 and costs[opt] >= cR - 1e-6)

print("\nInterpretation: the preferred deployment cap varies with the cost ratio;")
print("this sensitivity analysis is not a universal cap-selection rule.")
