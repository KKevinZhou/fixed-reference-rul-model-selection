"""Public-distribution entry point: no NASA raw data, eight declared weights."""
from pathlib import Path
from argparse import Namespace
from datetime import datetime, timezone
from types import SimpleNamespace
import argparse
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'archive/source'))
from lib import portable_core as core
core.ROOT = ROOT
core.ARCHIVE = ARCHIVE = ROOT/'archive'
read, sha, write = core.read, core.sha, core.write
BIB_SHA = core.BIB_SHA


def verify():
    manifest = read(ROOT/'PUBLIC_MANIFEST.json')
    if sha(ROOT/'reference.bib') != BIB_SHA or manifest['bib_sha256'] != BIB_SHA:
        raise ValueError('Protected Bib changed')
    for row in manifest['files']:
        path = core.contained(row['path'])
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError(f'Public package integrity failure: {row["path"]}')
    jobs = read(ROOT/'replay_jobs.json')['jobs']
    if len(jobs) != 210 or sum(len(j['seeds']) for j in jobs) != 320:
        raise ValueError('Recorded successful-fit inventory differs')
    actual_weights = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*.pth')}
    selected_weights = {r['path'] for r in manifest['selected_checkpoints']}
    if actual_weights != selected_weights or len(selected_weights) != 8:
        raise ValueError('Selected checkpoint inventory differs')
    forbidden = [p for p in ROOT.rglob('*') if p.is_file() and
                 (p.suffix in ('.npz', '.pkl') or p.name.startswith(('train_FD00', 'test_FD00', 'RUL_FD00')) and p.suffix == '.txt')]
    if forbidden:
        raise ValueError('NASA raw/feature files are not part of this release')
    print(json.dumps({'status':'passed', 'files':len(manifest['files']), 'recorded_fits':320,
                      'recorded_checkpoints':640, 'included_checkpoints':8,
                      'NASA_raw_files_included':False, 'repository_publication_verified':False}, indent=2))
    return manifest


def verify_test_truth(dataset, targets):
    """Check labels already in derived predictions without distributing raw TXT."""
    values = [float(x) for x in targets]
    payload = {'units': list(range(1, len(values)+1)), 'original_targets': values}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    expected = read(ARCHIVE/'source/canonical_cmapss_sha256.json')['ordered_test_targets'][dataset]
    if len(values) != expected['test_unit_count'] or digest != expected['ordered_unit_target_sha256']:
        raise ValueError('Derived target payload differs from canonical NASA anchor')


def validate_result(path, spec=None, plan=None):
    path = core.resolve_history(path)
    result = read(path)
    audit, summary, protocol = core.scientific_modules()
    audit.recompute_validation_metrics(result, path)
    if spec is not None:
        if result['dataset'] != spec['dataset'] or result['model_seeds'] != spec['model_seeds']:
            raise ValueError('Planned result identity differs')
        for field in ('train_cap','reporting_cap','split_seed','endpoint_seed'):
            if result[field] != spec['arguments'][field]: raise ValueError(f'Planned {field} differs')
        expected = {k:v for k,v in spec['arguments'].items() if k in result['config']}
        if not protocol.config_matches(result['config'], expected): raise ValueError('Model configuration differs')
        if result['test_evaluated'] != (spec['stage']=='five_seed_test'): raise ValueError('Test phase differs')
        if result['new_execution_provenance']['numerical_policy'] != spec['numerical_policy']: raise ValueError('Precision policy differs')
    selected = {r['path'] for r in read(ROOT/'PUBLIC_MANIFEST.json')['selected_checkpoints']}
    recorded, checked = 0, 0
    for seed in result['per_seed_validation']:
        if seed['nonfinite_batches'] != 0: raise ValueError('Non-finite successful record')
        for objective, detail in seed['checkpoints'].items():
            recorded += 1
            weight = path.parent.parent/'checkpoints'/result['campaign_id']/result['dataset']/f'{result["tag"]}_seed{seed["seed"]}_{objective}.pth'
            relative = weight.relative_to(ROOT).as_posix()
            if relative in selected:
                if sha(weight) != detail['checkpoint_sha256']: raise ValueError('Selected checkpoint differs')
                checked += 1
            elif weight.exists():
                raise ValueError('Unselected checkpoint unexpectedly present')
    if result['test_evaluated']:
        for objective in ('self_scaled','fixed_reference'):
            entries = result['test_results'][objective]
            for row in [*entries['per_seed'], entries['ensemble']]:
                summary.validated_prediction_arrays(row, result['train_cap'], str(path))
                verify_test_truth(result['dataset'], row['original_targets'])
    return {'fit_count':len(result['model_seeds']), 'recorded_checkpoint_count':recorded,
            'verified_included_checkpoint_count':checked, 'result_sha256':sha(path)}


def recompute(output):
    verify()
    output = Path(output).resolve()
    if output.is_relative_to(ROOT): raise ValueError('Write output outside release')
    if output.exists() and any(output.iterdir()): raise FileExistsError('Use an empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    values = [validate_result(core.contained(j['result'])) for j in read(ROOT/'replay_jobs.json')['jobs']]
    counts = {key:sum(r[key] for r in values) for key in ('fit_count','recorded_checkpoint_count','verified_included_checkpoint_count')}
    if counts != {'fit_count':320,'recorded_checkpoint_count':640,'verified_included_checkpoint_count':8}:
        raise ValueError(counts)
    statistics = core.recompute_statistics(output)
    audit, summary, protocol = core.scientific_modules()
    adapter = SimpleNamespace(ROOT=ARCHIVE, SOURCE=ARCHIVE/'source', CAMPAIGN=ARCHIVE/'new_cap_campaign',
        OUTPUT=ARCHIVE/'continuation_20261011_v2', PLAN=ARCHIVE/'continuation_20261011_v2/immutable_plan.json',
        BIB=ROOT/'reference.bib', BIB_SHA=BIB_SHA, DATA=None,
        read=lambda p:read(core.resolve_history(p)),
        sha=lambda p:sha(Path(p) if Path(p).resolve().is_relative_to(output) else core.resolve_history(p)),
        atomic=write, utc=lambda:datetime.now(timezone.utc).isoformat(),
        validate=validate_result, summary_tools=summary, protocol=protocol, verify_test_truth=verify_test_truth)
    sys.modules['continue_reproduction_20261011'] = adapter
    spec = importlib.util.spec_from_file_location('public_results_generator', ROOT/'analysis/generate_public_results.py')
    generator = importlib.util.module_from_spec(spec); spec.loader.exec_module(generator)
    generator.REPRO=ARCHIVE; generator.SUMMARY=output/'summary'; generator.MANUSCRIPT=output/'manuscript'
    (generator.MANUSCRIPT/'figures').mkdir(parents=True)
    (generator.MANUSCRIPT/'reference.bib').write_bytes((ROOT/'reference.bib').read_bytes())
    generator.main()
    import numpy as np
    import pandas as pd
    matches = []
    for path in sorted((ROOT/'expected_summary').glob('*.csv')):
        actual, expected = pd.read_csv(generator.SUMMARY/path.name), pd.read_csv(path)
        if actual.shape != expected.shape or list(actual.columns) != list(expected.columns): raise ValueError('Table shape differs')
        for field in expected:
            if field in ('result_path','result_paths','lock_path','robustness_summary_path'): continue
            if pd.api.types.is_numeric_dtype(expected[field]):
                np.testing.assert_allclose(actual[field], expected[field], rtol=1e-9, atol=1e-9, equal_nan=True)
            elif actual[field].fillna('').astype(str).tolist() != expected[field].fillna('').astype(str).tolist():
                raise ValueError(f'Table values differ: {path.name}:{field}')
        matches.append(path.name)
    receipt = {'status':'passed', **counts, **statistics, 'summary_csvs_matched':matches,
               'NASA_raw_data_read':False, 'omitted_weights_verified':False,
               'historical_records_modified':False, 'new_training':False}
    write(output/'PUBLIC_RECOMPUTATION_RECEIPT.json', receipt)
    print(json.dumps(receipt, indent=2))


def inference(output, data_root):
    verify()
    import numpy as np
    import torch
    import confirmatory_rul_train as tr
    output = Path(output).resolve()
    if output.is_relative_to(ROOT): raise ValueError('Write receipt outside release')
    if output.exists(): raise FileExistsError('Receipt already exists')
    data_root = Path(data_root).resolve()
    canonical = read(ARCHIVE/'source/canonical_cmapss_sha256.json')
    lock = read(ARCHIVE/'new_cap_campaign/final_cap_decisions.json')
    checks=[]
    for dataset in ('FD001','FD002','FD003','FD004'):
        paths=tr.raw_paths(data_root,dataset)
        for path in paths.values():
            if sha(path) != canonical['files'][path.name]: raise ValueError('External NASA file differs')
        raw=tr.read_raw_training(paths['train'])
        training_units,_=tr.split_units(raw,20260724,0.15)
        pre=tr.FoldPreprocessor(multi_condition=tr.DEFAULTS[dataset]['multi_condition'],random_state=20260724).fit(raw[raw.unit.isin(training_units)].reset_index(drop=True))
        test,truth,targets=tr.read_raw_test(paths['test'],paths['truth'])
        frame=pre.transform(test)
        for cap,objective,seed in [(90,'self_scaled',45),(lock['decisions'][dataset]['fixed_reference']['selected_cap'],'fixed_reference',49)]:
            tag=f'cap_{dataset}_{cap}'.lower(); base=ARCHIVE/'continuation_20261011_v2/five_seed_test'
            result=read(base/'results'/f'{dataset}_{tag}.json'); config=Namespace(**result['config'])
            rows=tr.WindowDataset(frame,config.seq_len,targets,augment=False,stride=1)
            loader=tr.make_loader(rows,config.batch*2,shuffle=False,seed=0)
            weight=base/'checkpoints'/result['campaign_id']/dataset/f'{tag}_seed{seed}_{objective}.pth'
            tr.set_seed(seed); model=tr.build_model(config)
            model.load_state_dict(torch.load(weight,map_location=tr.DEVICE,weights_only=True))
            predicted=tr.endpoint_predictions(model,loader,config.ema_span)
            archived=next(r for r in result['test_results'][objective]['per_seed'] if r['seed']==seed)
            actual=np.asarray([predicted[u] for u in archived['units']]); expected=np.asarray(archived['raw_predictions'])
            np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=1e-6)
            checks.append({'dataset':dataset,'cap':cap,'objective':objective,'seed':seed,'outputs':len(actual),'max_abs_difference':float(np.max(np.abs(actual-expected)))})
            del model
    receipt={'status':'passed','included_checkpoints':8,'outputs_checked':sum(r['outputs'] for r in checks),'checks':checks,'external_NASA_input_required':True}
    write(output,receipt); print(json.dumps(receipt,indent=2))


core.verify = verify
core.validate_result = validate_result


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('verify'); sub.add_parser('list-jobs')
    q=sub.add_parser('recompute'); q.add_argument('--output',required=True)
    q=sub.add_parser('inference-check'); q.add_argument('--output',required=True); q.add_argument('--data-root',required=True)
    q=sub.add_parser('train'); q.add_argument('--job',required=True); q.add_argument('--output',required=True); q.add_argument('--data-root',required=True); q.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.command=='verify':verify()
    elif a.command=='list-jobs':
        for j in read(ROOT/'replay_jobs.json')['jobs']:print(j['id'],j['dataset'],j['seeds'],j['numerical_policy'])
    elif a.command=='recompute':recompute(a.output)
    elif a.command=='inference-check':inference(a.output,a.data_root)
    else:core.train(a)


if __name__=='__main__':main()
