"""New replay of frozen trainer with raw prediction capture and declared precision."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '2')
sys.path[:0] = [str(ROOT/'vendor'), str(ROOT/'archive/source'), str(ROOT/'archive')]
import confirmatory_rul_train as frozen
import final_attested_trainer as capture
from attention_precision_guard import install_attention_precision_guard


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--numerical-policy', choices=['original_amp_float16', 'original_amp_with_fp32_attention'], required=True)
    options, tail = parser.parse_known_args()
    sys.argv = [str(ROOT/'archive/source/confirmatory_rul_train.py'), *tail]
    args = frozen.parse_args()
    frozen.FoldPreprocessor.__module__ = '__main__'
    globals()['FoldPreprocessor'] = frozen.FoldPreprocessor
    if options.numerical_policy == 'original_amp_with_fp32_attention':
        if args.carrier != 'dualstream' or not args.validation_only:
            raise ValueError('FP32 attention is a separate validation-only carrier sensitivity')
        original_build = frozen.build_model
        def build(config):
            model = original_build(config)
            install_attention_precision_guard(model)
            return model
        frozen.build_model = build
    original_evaluate = frozen.evaluate_test
    def evaluate(*values, **kwargs):
        captured = []
        original_metric = frozen.metric_bundle
        def metric(*args, **kwargs):
            bundle = original_metric(*args, **kwargs)
            captured.append(bundle)
            return bundle
        frozen.metric_bundle = metric
        try:
            result = original_evaluate(*values, **kwargs)
        finally:
            frozen.metric_bundle = original_metric
        return capture.attach_per_seed_predictions(result, captured, args.train_cap, args.reporting_cap)
    frozen.evaluate_test = evaluate
    print(json.dumps({'new_training': True, 'historical_recovery': False, 'numerical_policy': options.numerical_policy}), flush=True)
    frozen.main()


if __name__ == '__main__': main()
