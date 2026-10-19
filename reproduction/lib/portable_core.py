"""Portable entry points for the real October 2026 rerun; never rewrite evidence."""
from __future__ import annotations
import argparse
from argparse import Namespace
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'archive'
BIB_SHA = 'c5669ec88d6185fdea2bc86693ca880185754aa546288cecd73cd7f15a58dc36'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path[:0] = [str(ROOT/'vendor'), str(ARCHIVE/'source')]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def contained(relative):
    path = (ROOT/relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Package path escapes root: {relative}')
    return path


def resolve_history(path):
    """Resolve recorded paths only through the explicit, hashed relocation map."""
    original = str(path)
    candidate = Path(path).resolve()
    if candidate.is_relative_to(ROOT) and candidate.exists():
        return candidate
    key = original.replace('\\', '/').lower()
    mapping = read(ROOT/'historical_path_map.json')
    if key not in mapping:
        raise FileNotFoundError(f'No packaged historical binding: {original}')
    return contained(mapping[key])


def verify():
    manifest = read(ROOT/'PACKAGE_MANIFEST.json')
    if manifest.get('bib_sha256') != BIB_SHA or sha(ROOT/'reference.bib') != BIB_SHA:
        raise ValueError('Protected user bibliography changed')
    for row in manifest['files']:
        path = contained(row['path'])
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError(f'Package integrity failure: {row["path"]}')
    jobs = read(ROOT/'replay_jobs.json')['jobs']
    if len(jobs) != 210 or sum(len(j['seeds']) for j in jobs) != 320:
        raise ValueError('Successful fit inventory differs')
    if sum(r['path'].endswith('.pth') for r in manifest['files']) != 640:
        raise ValueError('Committed checkpoint inventory differs')
    result = {'status': 'passed', 'files_checked': len(manifest['files']),
              'archive_records': len(jobs),
              'successful_fits': 320, 'distinct_checkpoint_files': 640,
              'public_release': False, 'historical_outputs_recovered': False}
    print(json.dumps(result, indent=2))
    return result


def scientific_modules():
    import preflight_confirmatory as audit
    import summarize_confirmatory as summary
    import run_confirmatory_final as protocol
    return audit, summary, protocol


def validate_result(path, spec=None, plan=None):
    path = resolve_history(path)
    result = read(path)
    audit, summary, protocol = scientific_modules()
    audit.recompute_validation_metrics(result, path)
    if spec is not None:
        if result['dataset'] != spec['dataset'] or result['model_seeds'] != spec['model_seeds']:
            raise ValueError('Result identity differs from plan')
        for key in ('train_cap', 'reporting_cap', 'split_seed', 'endpoint_seed'):
            if result[key] != spec['arguments'][key]:
                raise ValueError(f'Configuration differs: {key}')
        if result['test_evaluated'] != (spec['stage'] == 'five_seed_test'):
            raise ValueError('Test/validation separation differs')
        expected_config = {key: value for key, value in spec['arguments'].items() if key in result['config']}
        if not protocol.config_matches(result['config'], expected_config):
            raise ValueError('Model configuration differs')
        if result['new_execution_provenance']['numerical_policy'] != spec['numerical_policy']:
            raise ValueError('Numerical policy differs')
    count = 0
    for seed in result['per_seed_validation']:
        if seed['nonfinite_batches'] != 0:
            raise ValueError('Successful result has non-finite batches')
        for objective, detail in seed['checkpoints'].items():
            weight = path.parent.parent/'checkpoints'/result['campaign_id']/result['dataset']/f'{result["tag"]}_seed{seed["seed"]}_{objective}.pth'
            if sha(weight) != detail['checkpoint_sha256']:
                raise ValueError(f'Checkpoint digest differs: {weight}')
            count += 1
    if result['test_evaluated']:
        for objective in ('self_scaled', 'fixed_reference'):
            entries = result['test_results'][objective]
            for row in [*entries['per_seed'], entries['ensemble']]:
                summary.validated_prediction_arrays(row, result['train_cap'], str(path))
    return {'result_sha256': sha(path), 'fit_count': len(result['model_seeds']), 'verified_checkpoint_count': count}


def selected_methods():
    plan = read(ARCHIVE/'continuation_20261011_v2/immutable_plan.json')
    lock = read(ARCHIVE/'new_cap_campaign/final_cap_decisions.json')
    output = {}
    for dataset in ('FD001', 'FD002', 'FD003', 'FD004'):
        methods = {}
        for family in ('cap', 'tpe'):
            if family == 'tpe' and dataset not in ('FD001', 'FD002'):
                continue
            for objective in ('self_scaled', 'fixed_reference', 'fixed_cap_125'):
                key = f'cap_{dataset}_{int(lock["decisions"][dataset][objective]["selected_cap"])}' if family == 'cap' else f'tpe_{dataset}_{objective}'
                result = read(resolve_history(plan['jobs'][key]['result_path']))
                methods[family+'_'+objective] = result['test_results']['self_scaled' if objective == 'self_scaled' else 'fixed_reference']
        output[dataset] = methods
    return output


def recompute_statistics(destination):
    import numpy as np
    import pandas as pd
    _, tools, _ = scientific_modules()
    expected_methods = pd.read_csv(ARCHIVE/'continuation_20261011_v2/verified_exports/test_selected_methods.csv').set_index(['dataset', 'method'])
    expected_pairs = pd.read_csv(ARCHIVE/'continuation_20261011_v2/verified_exports/test_paired_contrasts.csv').set_index(['dataset', 'comparison'])
    pairs, methods = [], []
    for dataset, entries in selected_methods().items():
        for name, value in entries.items():
            metrics = []
            for row in value['per_seed']:
                pr = np.clip(np.asarray(row['raw_predictions'], dtype=float), 0, 125)
                y = np.asarray(row['original_targets'], dtype=float)
                np.testing.assert_allclose(pr, row['common_support_predictions'], rtol=1e-12, atol=1e-12)
                metric = tools.rmse(pr, np.minimum(y, 125))
                np.testing.assert_allclose(metric, row['reference_rmse'], rtol=1e-10, atol=1e-10)
                metrics.append(metric)
            e = value['ensemble']
            target = np.minimum(e['original_targets'], 125)
            point = tools.rmse(e['common_support_predictions'], target)
            low, high = tools.single_engine_bootstrap_rmse(e['common_support_predictions'], target, f'{dataset}:{name}', resamples=2000)
            row = {'dataset': dataset, 'method': name, 'seed_rmse_mean': float(np.mean(metrics)),
                   'seed_rmse_std': float(np.std(metrics, ddof=1)), 'ensemble_rmse': point,
                   'engine_bootstrap_low': low, 'engine_bootstrap_high': high}
            for field, val in row.items():
                if field not in ('dataset', 'method'):
                    np.testing.assert_allclose(val, expected_methods.loc[(dataset, name), field], rtol=1e-10, atol=1e-10)
            methods.append(row)
        contrasts = [('fixed_vs_self_cap_grid', 'cap_fixed_reference', 'cap_self_scaled'),
                     ('fixed_cap_grid_vs_fixed125', 'cap_fixed_reference', 'cap_fixed_cap_125')]
        if dataset in ('FD001', 'FD002'):
            contrasts += [('fixed_vs_self_tpe', 'tpe_fixed_reference', 'tpe_self_scaled'),
                          ('fixed_reference_tpe_vs_fixed_cap_125_tpe', 'tpe_fixed_reference', 'tpe_fixed_cap_125')]
        for label, left_name, right_name in contrasts:
            left, right = entries[left_name], entries[right_name]
            if [r['seed'] for r in left['per_seed']] != [r['seed'] for r in right['per_seed']]:
                raise ValueError('Paired model seeds differ')
            differences = np.array([l['reference_rmse']-r['reference_rmse'] for l, r in zip(left['per_seed'], right['per_seed'], strict=True)])
            statistic, pvalue, count = tools.exact_signed_rank_test(differences)
            le, re = left['ensemble'], right['ensemble']
            if le['units'] != re['units'] or le['original_targets'] != re['original_targets']:
                raise ValueError('Paired engine targets differ')
            point, low, high, negative = tools.paired_engine_bootstrap(le['common_support_predictions'], re['common_support_predictions'], np.minimum(le['original_targets'], 125), f'{dataset}:{label}', resamples=10000)
            row = {'dataset': dataset, 'comparison': label, 'mean_difference': float(differences.mean()),
                   'wilcoxon_statistic': statistic, 'wilcoxon_pvalue': pvalue, 'nonzero_pairs': count,
                   'ensemble_difference': point, 'paired_engine_ci_low': low, 'paired_engine_ci_high': high,
                   'bootstrap_fraction_negative': negative}
            pairs.append(row)
    tools.add_holm_adjustment(pairs, 'holm_all12_pvalue')
    tools.add_holm_adjustment([r for r in pairs if r['comparison'] == 'fixed_vs_self_cap_grid'], 'holm_primary4_pvalue')
    tools.add_holm_adjustment([r for r in pairs if r['comparison'] != 'fixed_vs_self_cap_grid'], 'holm_secondary8_pvalue')
    for row in pairs:
        for field, value in row.items():
            if field not in ('dataset', 'comparison'):
                np.testing.assert_allclose(value, expected_pairs.loc[(row['dataset'], row['comparison']), field], rtol=1e-10, atol=1e-10)
    pd.DataFrame(methods).to_csv(destination/'independent_method_statistics.csv', index=False)
    pd.DataFrame(pairs).to_csv(destination/'independent_paired_statistics.csv', index=False)
    return {'methods_recomputed': len(methods), 'paired_contrasts_recomputed': len(pairs),
            'paired_engine_bootstrap_resamples': 10000, 'single_engine_bootstrap_resamples': 2000}


def recompute(output):
    verify()
    output = Path(output).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError('Write recomputation output outside the immutable package')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new, empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for job in read(ROOT/'replay_jobs.json')['jobs']:
        results.append(validate_result(contained(job['result'])))
    counts = {'fits': sum(r['fit_count'] for r in results), 'checkpoints': sum(r['verified_checkpoint_count'] for r in results)}
    if counts != {'fits': 320, 'checkpoints': 640}:
        raise ValueError(counts)
    statistics = recompute_statistics(output)
    audit, summary, protocol = scientific_modules()
    adapter = SimpleNamespace()
    adapter.__dict__.update(ROOT=ARCHIVE, SOURCE=ARCHIVE/'source', CAMPAIGN=ARCHIVE/'new_cap_campaign',
        DATA=ROOT/'data_root', OUTPUT=ARCHIVE/'continuation_20261011_v2',
        PLAN=ARCHIVE/'continuation_20261011_v2/immutable_plan.json',
        BIB=ROOT/'reference.bib', BIB_SHA=sha(ROOT/'reference.bib'),
        read=lambda p: read(resolve_history(p)),
        sha=lambda p: sha(Path(p) if Path(p).resolve().is_relative_to(output) else resolve_history(p)),
        atomic=write, utc=lambda: datetime.now(timezone.utc).isoformat(),
        validate=validate_result, summary_tools=summary, protocol=protocol)
    sys.modules['continue_reproduction_20261011'] = adapter
    generator_path = ROOT/'analysis/generate_verified_manuscript_results.py'
    spec = importlib.util.spec_from_file_location('portable_manuscript_generator', generator_path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    generator.REPRO = ARCHIVE
    generator.SUMMARY = output/'summary'
    generator.MANUSCRIPT = output/'manuscript'
    (generator.MANUSCRIPT/'figures').mkdir(parents=True)
    (generator.MANUSCRIPT/'reference.bib').write_bytes((ROOT/'reference.bib').read_bytes())
    generator.main()
    import numpy as np
    import pandas as pd
    compared = []
    for expected_path in (ROOT/'expected_summary').glob('*.csv'):
        actual = pd.read_csv(generator.SUMMARY/expected_path.name)
        expected = pd.read_csv(expected_path)
        if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
            raise ValueError(f'Summary shape differs: {expected_path.name}')
        for field in expected.columns:
            if field in ('result_path', 'result_paths', 'lock_path', 'robustness_summary_path'):
                continue
            if pd.api.types.is_numeric_dtype(expected[field]):
                np.testing.assert_allclose(actual[field], expected[field], rtol=1e-9, atol=1e-9, equal_nan=True)
            elif actual[field].fillna('').astype(str).tolist() != expected[field].fillna('').astype(str).tolist():
                raise ValueError(f'Summary text differs: {expected_path.name}:{field}')
        compared.append(expected_path.name)
    receipt = {'status': 'passed', 'root': str(ROOT), **counts, **statistics,
               'summary_csvs_matched': compared, 'historical_record_bytes_changed': False,
               'archive_record_paths_are_relocated_only_at_read_time': True,
               'new_training_performed': False}
    write(output/'RECOMPUTATION_RECEIPT.json', receipt)
    print(json.dumps(receipt, indent=2))


def train(options):
    """Replay a recorded configuration into new outputs, never historical receipts."""
    verify()
    jobs = read(ROOT/'replay_jobs.json')['jobs']
    job = next((r for r in jobs if r['id'] == options.job), None)
    if job is None:
        raise ValueError('Unknown job ID; run list-jobs')
    record = read(contained(job['result']))
    output = Path(options.output).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError('Training output must be outside immutable package')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new empty training output directory')
    data = Path(options.data_root or ROOT/'data_root').resolve()
    canonical = read(ARCHIVE/'source/canonical_cmapss_sha256.json')
    for name in (f'train_{record["dataset"]}.txt', *([] if record['validation_only'] else [f'test_{record["dataset"]}.txt', f'RUL_{record["dataset"]}.txt'])):
        path = data/'Experiment_FD001/CMAPSSData'/name
        if sha(path) != canonical['files'][name]:
            raise ValueError(f'NASA raw data differs: {name}')
    config = record['config']
    args = {k: config[k] for k in ('seq_len', 'lr', 'batch', 'epochs', 'patience', 'huber_delta', 'ema_span', 'train_stride', 'validation_interval', 'val_fraction', 'carrier')}
    seeds = record['model_seeds']
    if seeds != list(range(seeds[0], seeds[0]+len(seeds))):
        raise ValueError('Non-contiguous archived seeds require explicit individual replay')
    args.update(dataset=record['dataset'], data_root=data, output_root=output,
        campaign_id='portable-new-run', tag=job['id'], train_cap=record['train_cap'],
        reporting_cap=record['reporting_cap'], split_seed=record['split_seed'],
        endpoint_seed=record['endpoint_seed'], seeds=len(seeds), seed_base=seeds[0], bootstrap_seed=20260724)
    tail = [part for key, value in args.items() for part in ('--'+key, str(value))]
    if record['validation_only']:
        tail.append('--validation_only')
    command = [sys.executable, str(ROOT/'portable_train.py'), '--numerical-policy', job['numerical_policy'], *tail]
    if options.dry_run:
        print(json.dumps({'job': job, 'command': command}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    write(output/'NEW_RUN_PLAN.json', {'command': command, 'archived_job': job,
        'source_record_sha256': sha(contained(job['result'])),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'historical_recovery': False, 'new_training': True,
        'scope': 'recorded-configuration replay, not a new adaptive HPO selection or unseen validation'})
    import subprocess
    with (output/'training.log').open('x', encoding='utf-8') as log:
        code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    write(output/'RUN_STATUS.json', {'returncode': code, 'complete': code == 0})
    if code:
        raise RuntimeError(f'New training failed; original log retained: {output}')


def inference_check(output):
    verify()
    import numpy as np
    import pandas as pd
    import torch
    import confirmatory_rul_train as tr
    output = Path(output).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError('Write inference output outside the immutable package')
    if output.exists():
        raise FileExistsError('Use a new inference receipt path')
    lock = read(ARCHIVE/'new_cap_campaign/final_cap_decisions.json')
    checks = []
    for dataset in ('FD001', 'FD002', 'FD003', 'FD004'):
        npz = ARCHIVE/f'reconstructed/folds/{dataset}_split20260724/test_processed_rows.npz'
        with np.load(npz, allow_pickle=False) as values:
            frame = pd.DataFrame(values['test_features'], columns=values['features'].tolist())
            frame['unit'], frame['cycle'] = values['test_units'], values['test_cycles']
            targets = dict(zip(values['target_units'].tolist(), values['target_original_rul'].tolist(), strict=True))
        for cap, objective, seed in [(90, 'self_scaled', 45), (lock['decisions'][dataset]['fixed_reference']['selected_cap'], 'fixed_reference', 49)]:
            tag = f'cap_{dataset}_{cap}'.lower()
            result_root = ARCHIVE/'continuation_20261011_v2/five_seed_test'
            result = read(result_root/'results'/f'{dataset}_{tag}.json')
            config = Namespace(**result['config'])
            rows = tr.WindowDataset(frame, config.seq_len, targets, augment=False, stride=1)
            loader = tr.make_loader(rows, config.batch*2, shuffle=False, seed=0)
            weight = result_root/'checkpoints'/result['campaign_id']/dataset/f'{tag}_seed{seed}_{objective}.pth'
            checkpoint = next(r for r in result['per_seed_validation'] if r['seed'] == seed)['checkpoints'][objective]
            if sha(weight) != checkpoint['checkpoint_sha256']:
                raise ValueError('Inference checkpoint differs')
            tr.set_seed(seed)
            model = tr.build_model(config)
            model.load_state_dict(torch.load(weight, map_location=tr.DEVICE, weights_only=True))
            predicted = tr.endpoint_predictions(model, loader, config.ema_span)
            archived = next(r for r in result['test_results'][objective]['per_seed'] if r['seed'] == seed)
            actual = np.asarray([predicted[unit] for unit in archived['units']])
            expected = np.asarray(archived['raw_predictions'])
            np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-6)
            checks.append({'dataset': dataset, 'cap': cap, 'objective': objective, 'seed': seed,
                           'predictions': len(actual), 'max_abs_difference': float(np.max(np.abs(actual-expected)))})
            del model
    receipt = {'status': 'passed', 'checkpoints_checked': len(checks),
               'predictions_checked': sum(r['predictions'] for r in checks), 'checks': checks,
               'archive_files_modified': False}
    write(output, receipt)
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify')
    sub.add_parser('list-jobs')
    p = sub.add_parser('recompute'); p.add_argument('--output', required=True)
    p = sub.add_parser('inference-check'); p.add_argument('--output', required=True)
    p = sub.add_parser('train'); p.add_argument('--job', required=True); p.add_argument('--output', required=True)
    p.add_argument('--data-root'); p.add_argument('--dry-run', action='store_true')
    options = parser.parse_args()
    if options.command == 'verify': verify()
    elif options.command == 'list-jobs':
        for row in read(ROOT/'replay_jobs.json')['jobs']:
            print(row['id'], row['dataset'], row['seeds'], row['numerical_policy'])
    elif options.command == 'recompute': recompute(options.output)
    elif options.command == 'inference-check': inference_check(options.output)
    else: train(options)


if __name__ == '__main__':
    main()
