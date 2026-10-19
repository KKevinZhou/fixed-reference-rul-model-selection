import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_carrier_transfer as carrier
from run_carrier_transfer import selected_cap
import preflight_confirmatory as preflight


def test_selected_cap_minimizes_requested_metric():
    profile = [
        {"cap": 90, "loss": 10.0},
        {"cap": 100, "loss": 8.0},
        {"cap": 125, "loss": 9.0},
    ]
    assert selected_cap(profile, "loss") == 100


def test_selected_cap_prefers_larger_cap_on_exact_tie():
    profile = [
        {"cap": 90, "loss": 8.0},
        {"cap": 115, "loss": 8.0},
        {"cap": 125, "loss": 9.0},
    ]
    assert selected_cap(profile, "loss") == 115


def test_valid_result_rejects_metric_and_config_tampering(tmp_path):
    source_manifest = {
        "trainer_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "raw_train_sha256": {"FD001": "c" * 64},
    }
    result = {
        "campaign_id": "carrier-test",
        "dataset": "FD001",
        "train_cap": 90,
        "reporting_cap": 125.0,
        "split_seed": carrier.SPLIT_SEED,
        "endpoint_seed": carrier.ENDPOINT_SEED,
        "model_seeds": [carrier.MODEL_SEED],
        "validation_only": True,
        "test_evaluated": False,
        "config": {
            "carrier": "dualstream",
            "seq_len": 50,
            "lr": 5e-4,
            "batch": 128,
            "huber_delta": 1.0,
        },
        "validation_summary": {
            "self_scaled": {"own_rmse": {"values": [8.0]}},
            "fixed_reference": {
                "reference_rmse": {"values": [9.0]}
            },
        },
        "source_fingerprints": {
            "trainer_sha256": "a" * 64,
            "model_sha256": "b" * 64,
        },
        "data_fingerprints": {"train_sha256": "c" * 64},
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    kwargs = {
        "campaign_id": "carrier-test",
        "dataset": "FD001",
        "cap": 90,
        "source_manifest": source_manifest,
    }
    assert carrier.valid_result(path, **kwargs)

    changed = json.loads(json.dumps(result))
    changed["config"]["carrier"] = "compact_tcn"
    path.write_text(json.dumps(changed), encoding="utf-8")
    assert not carrier.valid_result(path, **kwargs)

    changed = json.loads(json.dumps(result))
    changed["validation_summary"]["fixed_reference"]["reference_rmse"][
        "values"
    ][0] = float("nan")
    path.write_text(json.dumps(changed), encoding="utf-8")
    assert not carrier.valid_result(path, **kwargs)


def test_carrier_check_rejects_existing_test_marker(tmp_path, monkeypatch):
    data = tmp_path / "data"
    campaign = tmp_path / "campaign"
    data.mkdir()
    campaign.mkdir()
    (campaign / "TEST_EVALUATION_STARTED.json").write_text(
        "{}", encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_carrier_transfer.py",
            "--data_root",
            str(data),
            "--campaign_root",
            str(campaign),
        ],
    )
    with pytest.raises(RuntimeError, match="must finish before test"):
        carrier.main()


def test_preflight_rebuilds_second_carrier_matrix(tmp_path, monkeypatch):
    datasets = ("FD001", "FD002")
    caps = (90, 100, 110, 115, 125)
    manifest = {
        "epochs": 40,
        "patience": 6,
        "train_stride": 5,
        "validation_interval": 2,
        "trainer_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "raw_train_sha256": {
            "FD001": "c" * 64,
            "FD002": "d" * 64,
        },
    }
    manifest_path = tmp_path / "campaign_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    robustness = tmp_path / "tpe_robustness_summary.json"
    robustness.write_text("{}", encoding="utf-8")
    result_dir = tmp_path / "carrier_transfer" / "results"
    result_dir.mkdir(parents=True)
    log_dir = tmp_path / "carrier_transfer" / "logs"
    log_dir.mkdir()
    hashes = {}
    selections = {}
    for dataset in datasets:
        profile = []
        for cap in caps:
            result = {
                "dataset": dataset,
                "train_cap": cap,
                "self": float(cap - 80),
                "self_ref": float(abs(cap - 115) + 9),
                "fixed": float(abs(cap - 125) + 8),
            }
            path = result_dir / f"{dataset}_cap{cap}.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            hashes[str(path.resolve())] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            profile.append(
                {
                    "cap": cap,
                    "self_scaled_own_rmse": result["self"],
                    "self_checkpoint_reference_rmse": result["self_ref"],
                    "fixed_reference_rmse": result["fixed"],
                }
            )
        selections[dataset] = {
            "self_scaled_selected_cap": 90,
            "fixed_reference_selected_cap": 125,
            "selection_changed": True,
            "profile": profile,
        }
    plan = {
        "protocol": "confirmatory-carrier-transfer-v1",
        "role": (
            "validation-only single-split, single-seed carrier sensitivity "
            "check; cannot change the locked cap-grid, TPE, or standard-test "
            "reporting decisions"
        ),
        "datasets": list(datasets),
        "caps": list(caps),
        "reporting_cap": 125,
        "split_seed": 20260724,
        "endpoint_seed": 21260727,
        "model_seed": 42,
        "carrier": "dualstream",
        "config": {
            "FD001": {"seq_len": 50, "lr": 5e-4},
            "FD002": {"seq_len": 50, "lr": 1e-3},
        },
        "epochs": 40,
        "patience": 6,
        "train_stride": 5,
        "validation_interval": 2,
        "selection_rule": (
            "minimum primary-split validation objective; exact ties prefer "
            "the larger cap"
        ),
        "primary_campaign_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "tpe_robustness_summary_sha256": hashlib.sha256(
            robustness.read_bytes()
        ).hexdigest(),
        "runner_sha256": preflight.sha256_file(
            ROOT / "run_carrier_transfer.py"
        ),
        "trainer_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "raw_train_sha256": manifest["raw_train_sha256"],
        "test_files_permitted": False,
    }
    plan["campaign_id"] = (
        "carrierv1-" + preflight.stable_hash(plan)[:16]
    )
    plan_path = tmp_path / "carrier_transfer_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    for dataset in datasets:
        for cap in caps:
            (
                log_dir
                / f"{dataset}_cap{cap}_{plan['campaign_id']}.log"
            ).write_text("completed\n", encoding="utf-8")
    summary = {
        "status": "complete",
        "protocol": plan["protocol"],
        "campaign_id": plan["campaign_id"],
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "selection_frozen": True,
        "selection_changed_count": 2,
        "selections": selections,
        "result_sha256": hashes,
        "test_evaluated": False,
    }
    (tmp_path / "carrier_transfer_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )

    def rebuild(_path, result, *_args):
        return {
            "self_scaled": {
                "own_rmse": {"mean": result["self"]},
                "reference_rmse": {"mean": result["self_ref"]},
            },
            "fixed_reference": {
                "reference_rmse": {"mean": result["fixed"]}
            },
        }

    monkeypatch.setattr(preflight, "validate_validation_result", rebuild)
    _, rebuilt, files = preflight.carrier_transfer_audit(
        tmp_path, manifest
    )
    assert rebuilt["selection_changed_count"] == 2
    assert len(files) == 22
