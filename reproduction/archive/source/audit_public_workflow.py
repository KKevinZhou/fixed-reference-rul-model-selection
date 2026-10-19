"""Verify the pinned public RULe source used by the manuscript audit.

The script downloads four text files from one immutable Git commit, verifies
their SHA-256 digests, and checks the exact source-line fragments supporting
the audit table.  It does not execute or redistribute the third-party code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


COMMIT = "4cf4563d1abcf350de9958a0de26d2d0126c819d"
BASE_URL = (
    "https://raw.githubusercontent.com/MariosKef/RULe/"
    f"{COMMIT}/RULe"
)
EXPECTED_SHA256 = {
    "main.py": "713c062f26569d96274fad52296b90034285ff81d992f1bd7a10f27f44f75bb3",
    "modeling.py": "87e8bd811343d0fcfbed15302d59ffa4bed23f3ffd60fcd4a5742412375b9e4d",
    "objective.py": "59ca6e267d035dcb482df5da2684b29c030806e5a95cbda921f4a1b3d16f4cde",
    "preprocessing.py": "ef4be844069f5db884b492e2840dd0a309dfd7c5bfb8bfc0124e74738fe1fa7f",
}
LINE_CHECKS = {
    "main.py": [
        {
            "range": [61, 64],
            "contains": [
                'OrdinalSpace([25, 75], "percentage")',
                'OrdinalSpace([110, 130], "rul")',
                'NominalSpace(["linear", "nonlinear"], "rul_style")',
            ],
        },
        {
            "range": [112, 129],
            "contains": ["* percentage", "* rul", "* rul_style"],
        },
    ],
    "objective.py": [
        {
            "range": [120, 142],
            "contains": [
                "train_x, train_y = build_data(",
                "test_x, test_y = build_data(",
                'net_cfg=net_cfg',
                'label=net_cfg["rul_style"]',
            ],
        },
        {
            "range": [287, 333],
            "contains": [
                'test_results_df["predicted_mu"]',
                'test_results_df["T"]',
                'return results["rmse_test"].mean()',
            ],
        },
    ],
    "preprocessing.py": [
        {
            "range": [63, 91],
            "contains": [
                'out_y.append(net_cfg["rul"])',
                '(0 - net_cfg["rul"])',
            ],
        },
    ],
    "modeling.py": [
        {
            "range": [84, 93],
            "contains": ["validation_data=(test_X, test_y)"],
        },
    ],
}


def validate_receipt(payload: dict) -> None:
    if payload.get("schema_version") != 1:
        raise RuntimeError("Unsupported public-workflow audit schema")
    if payload.get("repository") != "https://github.com/MariosKef/RULe":
        raise RuntimeError("Unexpected audited repository")
    if payload.get("commit") != COMMIT:
        raise RuntimeError("Audit receipt commit mismatch")
    observed = {
        row.get("path"): row.get("sha256")
        for row in payload.get("files", [])
        if isinstance(row, dict)
    }
    if observed != EXPECTED_SHA256:
        raise RuntimeError("Audit receipt source hashes do not match constants")
    if payload.get("test_or_truth_data_used") is not False:
        raise RuntimeError("Source audit must not use benchmark test outcomes")


def fetch_and_verify(timeout: int = 30) -> dict:
    rows = []
    for name, expected_hash in sorted(EXPECTED_SHA256.items()):
        url = f"{BASE_URL}/{name}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "RESS-reproducibility-audit/1.0"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read()
        observed_hash = hashlib.sha256(content).hexdigest()
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"Pinned source hash mismatch for {name}: {observed_hash}"
            )
        text = content.decode("utf-8")
        lines = text.splitlines()
        for check in LINE_CHECKS[name]:
            start, end = check["range"]
            excerpt = "\n".join(lines[start - 1 : end])
            for fragment in check["contains"]:
                if fragment not in excerpt:
                    raise RuntimeError(
                        f"{name}:{start}-{end} lacks {fragment!r}"
                    )
        rows.append(
            {
                "path": name,
                "url": url,
                "sha256": observed_hash,
                "line_checks": LINE_CHECKS[name],
            }
        )
    receipt = {
        "schema_version": 1,
        "repository": "https://github.com/MariosKef/RULe",
        "commit": COMMIT,
        "files": rows,
        "test_or_truth_data_used": False,
        "scope": (
            "Static source audit only; no third-party code was executed "
            "or redistributed."
        ),
    }
    validate_receipt(receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    observed = fetch_and_verify(args.timeout)
    if args.receipt is not None:
        stored = json.loads(args.receipt.read_text(encoding="utf-8"))
        validate_receipt(stored)
        if stored != observed:
            raise RuntimeError("Stored receipt differs from live pinned audit")
    print(json.dumps(observed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
