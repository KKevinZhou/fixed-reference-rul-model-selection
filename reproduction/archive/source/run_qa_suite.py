"""Run the full artifact test suite and write a hash-bound JUnit receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree


HERE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def source_hashes(root: Path) -> dict[str, str]:
    files = sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in files
    }


def junit_counts(path: Path) -> dict[str, int]:
    root = ElementTree.parse(path).getroot()
    suites = (
        [root]
        if root.tag.rsplit("}", 1)[-1] == "testsuite"
        else [
            child
            for child in root
            if child.tag.rsplit("}", 1)[-1] == "testsuite"
        ]
    )
    values = {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    values["passed"] = (
        values["tests"]
        - values["failures"]
        - values["errors"]
        - values["skipped"]
    )
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:\NASA_Lab\ress_confirmatory_v2"),
    )
    args = parser.parse_args()
    qa_dir = args.output_root.resolve() / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    junit = qa_dir / "pytest-junit.xml"
    manifest_path = qa_dir / "pytest_run_manifest.json"
    current_sources = source_hashes(HERE)
    if manifest_path.is_file() and junit.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("protocol") == "artifact-qa-receipt-v1"
            and existing.get("exit_code") == 0
            and existing.get("source_file_sha256") == current_sources
            and existing.get("junit_sha256") == sha256_file(junit)
            and existing.get("counts") == junit_counts(junit)
            and existing.get("environment_lock_sha256")
            == sha256_file(HERE / "environment-lock.txt")
            and existing.get("qa_runner_sha256")
            == sha256_file(Path(__file__).resolve())
        ):
            print(json.dumps(existing, indent=2))
            return
        if (
            args.output_root.resolve()
            / "PREMARKER_CAMPAIGN_INVENTORY.json"
        ).exists():
            raise RuntimeError(
                "QA inputs changed after the pre-marker inventory was frozen"
            )
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(HERE / "tests"),
        "-q",
        f"--junitxml={junit}",
    ]
    process = subprocess.run(command, cwd=HERE)
    if not junit.is_file() or junit.stat().st_size == 0:
        raise RuntimeError("pytest produced no JUnit report")
    counts = junit_counts(junit)
    payload = {
        "protocol": "artifact-qa-receipt-v1",
        "command": command,
        "cwd": str(HERE),
        "python": sys.version,
        "platform": platform.platform(),
        "exit_code": process.returncode,
        "counts": counts,
        "junit_file": junit.name,
        "junit_sha256": sha256_file(junit),
        "source_file_sha256": current_sources,
        "environment_lock_sha256": sha256_file(
            HERE / "environment-lock.txt"
        ),
        "qa_runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(manifest_path, payload)
    if (
        process.returncode != 0
        or counts["failures"] != 0
        or counts["errors"] != 0
        or counts["passed"] < 1
    ):
        raise RuntimeError(f"Artifact QA failed: {payload}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
