"""Plan-bound new execution using the unchanged trainer and raw-array capture.

This is not the historical official attestation shim. A precision guard, when
explicitly planned, is restricted to a separate carrier sensitivity campaign.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '2')
sys.path.insert(0, str(ROOT/'dependencies'))
sys.path.insert(0, str(ROOT/'source'))
import confirmatory_rul_train as frozen
import final_attested_trainer as capture_helper
from attention_precision_guard import install_attention_precision_guard


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--reproduction-plan', type=Path, required=True)
    parser.add_argument('--spec-key', required=True)
    options, tail = parser.parse_known_args()
    plan = json.loads(options.reproduction_plan.read_text(encoding='utf-8'))
    spec = plan['jobs'][options.spec_key]
    for path, digest in plan['source_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError(f'Planned source changed: {path}')
    sys.argv = [str(ROOT/'source'/'confirmatory_rul_train.py'), *tail]
    args = frozen.parse_args()
    expected_args = spec['arguments']
    for key, expected in expected_args.items():
        observed = getattr(args, key)
        if str(observed) != str(expected):
            raise RuntimeError(f'Argument differs from immutable plan: {key}: {observed} != {expected}')
    # Retain the original CLI's pickle class context for preprocessor hashes.
    frozen.FoldPreprocessor.__module__ = '__main__'
    globals()['FoldPreprocessor'] = frozen.FoldPreprocessor
    guarded_count = 0
    if spec['numerical_policy'] == 'original_amp_with_fp32_attention':
        if args.carrier != 'dualstream' or not args.validation_only:
            raise RuntimeError('Guarded carrier must remain separate and validation-only')
        original_build = frozen.build_model
        def build(config):
            nonlocal guarded_count
            model = original_build(config)
            guarded_count += install_attention_precision_guard(model)
            return model
        frozen.build_model = build
    elif spec['numerical_policy'] != 'original_amp_float16':
        raise RuntimeError('Unknown planned numerical policy')
    original_evaluate = frozen.evaluate_test
    def evaluate(*values, **kwargs):
        captured = []
        original_metric = frozen.metric_bundle
        def metric(*metric_args, **metric_kwargs):
            bundle = original_metric(*metric_args, **metric_kwargs)
            captured.append(bundle)
            return bundle
        frozen.metric_bundle = metric
        try:
            result = original_evaluate(*values, **kwargs)
        finally:
            frozen.metric_bundle = original_metric
        return capture_helper.attach_per_seed_predictions(result, captured, args.train_cap, args.reporting_cap)
    frozen.evaluate_test = evaluate
    original_atomic = frozen.atomic_json
    def write(path, value):
        if isinstance(value, dict) and 'per_seed_validation' in value:
            value['new_execution_provenance'] = {
                'protocol': plan['protocol'], 'plan_path': str(options.reproduction_plan.resolve()),
                'plan_file_sha256': sha(options.reproduction_plan), 'spec_key': options.spec_key,
                'capture_trainer_sha256': sha(__file__),
                'numerical_policy': spec['numerical_policy'], 'fp32_attention_modules': guarded_count,
                'historical_artifact_recovered': False, 'legacy_official_attestation': False,
                'per_seed_test_predictions_captured': bool(value.get('test_evaluated'))}
        return original_atomic(path, value)
    frozen.atomic_json = write
    print(json.dumps({'new_execution': True, 'spec_key': options.spec_key,
                      'numerical_policy': spec['numerical_policy']}), flush=True)
    frozen.main()


if __name__ == '__main__':
    main()
