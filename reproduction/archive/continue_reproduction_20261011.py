"""Finite continuation: separate guarded carrier and frozen five-seed test replay."""
from __future__ import annotations
import argparse
import hashlib
import json
import msvcrt
import os
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'source'
CAMPAIGN = ROOT/'new_cap_campaign'
DATA = ROOT.parents[1]/'organized_20261009'/'data_root'
OUTPUT = ROOT/'continuation_20261011_v2'
STATUS = OUTPUT/'status.json'
PLAN = OUTPUT/'immutable_plan.json'
BIB = Path(r'D:\paper\Fixed_Reference_RUL_Model_Selection\manuscript\reference.bib')
BIB_SHA = 'c5669ec88d6185fdea2bc86693ca880185754aa546288cecd73cd7f15a58dc36'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '2')
sys.path.insert(0, str(ROOT/'dependencies'))
sys.path.insert(0, str(SOURCE))
import run_confirmatory_final as protocol
import preflight_confirmatory as audit
import summarize_confirmatory as summary_tools


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temp, path)


def create_plan():
    if PLAN.exists():
        return read(PLAN)
    manifest = read(CAMPAIGN/'campaign_manifest.json')
    lock = read(CAMPAIGN/'final_cap_decisions.json')
    receipt = read(CAMPAIGN/'verified_exports'/'ALL_COMPLETED_VALIDATION_VERIFIED.json')
    assert receipt['verified_completed_fits'] == 205 and receipt['verified_checkpoint_count'] == 410
    assert sha(BIB) == BIB_SHA
    source_paths = [SOURCE/'confirmatory_rul_train.py', SOURCE/'baselines'/'models_v3.py',
                    SOURCE/'final_attested_trainer.py', SOURCE/'summarize_confirmatory.py',
                    ROOT/'summarize_new_test_replay.py', ROOT/'runtime_capture_trainer.py',
                    ROOT/'attention_precision_guard.py', Path(__file__)]
    selected_paths = [CAMPAIGN/'final_cap_decisions.json', CAMPAIGN/'campaign_manifest.json',
                      CAMPAIGN/'verified_exports'/'ALL_COMPLETED_VALIDATION_VERIFIED.json']
    jobs = {}
    def add(key, dataset, cap, config, stage, seeds, campaign_id, policy):
        out = OUTPUT/('guarded_carrier' if stage == 'guarded_carrier' else 'five_seed_test')
        tag = key.lower()
        args = {'dataset': dataset, 'data_root': str(DATA), 'output_root': str(out),
                'campaign_id': campaign_id, 'tag': tag, 'train_cap': float(cap),
                'reporting_cap': 125.0, 'split_seed': 20260724, 'endpoint_seed': 21260727,
                'seeds': len(seeds), 'seed_base': seeds[0], 'bootstrap_seed': 20260724,
                'validation_only': stage == 'guarded_carrier', **config}
        tail = []
        for name, value in args.items():
            if isinstance(value, bool):
                if value:
                    tail.append('--'+name)
            else:
                tail += ['--'+name, str(value)]
        jobs[key] = {'stage': stage, 'dataset': dataset, 'cap': cap, 'model_seeds': seeds,
                     'arguments': args, 'numerical_policy': policy,
                     'native_cli_arguments': tail, 'result_path': str(out/'results'/f'{dataset}_{tag}.json')}
    # First execute the real failing cell with the precision guard; then finish
    # the remaining profile under the same numerical policy in a separate root.
    carrier_order = [('FD002', 125)] + [(ds, cap) for ds in ['FD001', 'FD002'] for cap in [90, 100, 110, 115, 125] if (ds, cap) != ('FD002', 125)]
    for dataset, cap in carrier_order:
        config = protocol.expected_config(manifest, dataset, {'carrier': 'dualstream'})
        add(f'guard_{dataset}_cap{cap}', dataset, cap, config, 'guarded_carrier', [42],
            'guarded-carrier-'+sha(ROOT/'attention_precision_guard.py')[:16], 'original_amp_with_fp32_attention')
    for dataset in protocol.DATASETS:
        choices = lock['decisions'][dataset]
        caps = sorted({int(choices[name]['selected_cap']) for name in ('self_scaled', 'fixed_reference', 'fixed_cap_125')})
        for cap in caps:
            add(f'cap_{dataset}_{cap}', dataset, cap, protocol.expected_config(manifest, dataset),
                'five_seed_test', [45, 46, 47, 48, 49], manifest['campaign_id'], 'original_amp_float16')
    for dataset, objective in protocol.TPE_STUDIES:
        path = CAMPAIGN/'tpe_locks'/f'{dataset}_{objective}.json'
        selected_paths.append(path)
        tpe = read(path)
        params = tpe['best_params']
        config = protocol.expected_config(manifest, dataset, {key: params[key] for key in ('lr', 'batch', 'huber_delta')})
        add(f'tpe_{dataset}_{objective}', dataset, params['train_cap'], config, 'five_seed_test',
            [45, 46, 47, 48, 49], tpe['campaign_id'], 'original_amp_float16')
    canonical = read(SOURCE/'canonical_cmapss_sha256.json')
    raw = DATA/'Experiment_FD001'/'CMAPSSData'
    raw_hashes = {str(raw/name): digest for name, digest in canonical['files'].items()}
    for path, digest in raw_hashes.items():
        assert sha(Path(path)) == digest, path
    plan = {'protocol': 'new-execution-reproduction-plan-v1', 'run_id': str(uuid.uuid4()),
            'created_utc': utc(), 'selection_frozen_before_this_test_replay': True,
            'disclosed_prior_standard_test_contact': True, 'legacy_official_attestation': False,
            'scope': 'new actual execution of frozen current-method choices; separate numerically guarded carrier sensitivity',
            'source_sha256': {str(path): sha(path) for path in source_paths},
            'selection_input_sha256': {str(path): sha(path) for path in selected_paths},
            'raw_data_sha256': raw_hashes, 'protected_bib_sha256': BIB_SHA,
            'original_full_qa': read(ROOT/'qa'/'test_summary.json'),
            'original_carrier_terminal_failure_preserved': True,
            'analysis_plan': {'paired_engine_resamples': 10000, 'single_engine_resamples': 2000,
                              'seed_test': 'two-sided exact signed rank, zeros dropped, average tied ranks',
                              'holm': 'primary four, secondary eight, sensitivity all twelve',
                              'inference_scope': 'conditional descriptive reproduction due to prior test contact'},
            'jobs': jobs}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with PLAN.open('x', encoding='utf-8') as handle:
        json.dump(plan, handle, indent=2)
    return plan


def validate(path, spec, plan):
    result = read(path)
    assert result['dataset'] == spec['dataset'] and result['model_seeds'] == spec['model_seeds']
    expected_config = {key: spec['arguments'][key] for key in protocol.expected_config(read(CAMPAIGN/'campaign_manifest.json'), spec['dataset'])}
    assert protocol.config_matches(result['config'], expected_config), (result['config'], expected_config)
    assert result['test_evaluated'] == (spec['stage'] == 'five_seed_test')
    provenance = result['new_execution_provenance']
    assert provenance['plan_file_sha256'] == sha(PLAN)
    assert provenance['numerical_policy'] == spec['numerical_policy']
    assert result['source_fingerprints']['trainer_sha256'] == plan['source_sha256'][str(SOURCE/'confirmatory_rul_train.py')]
    assert result['source_fingerprints']['model_sha256'] == plan['source_sha256'][str(SOURCE/'baselines'/'models_v3.py')]
    audit.recompute_validation_metrics(result, path)
    checkpoint_count = 0
    for seed in result['per_seed_validation']:
        assert seed['nonfinite_batches'] == 0
        for objective, value in seed['checkpoints'].items():
            weight = path.parent.parent/'checkpoints'/result['campaign_id']/result['dataset']/f"{result['tag']}_seed{seed['seed']}_{objective}.pth"
            assert weight.exists() and sha(weight) == value['checkpoint_sha256']
            checkpoint_count += 1
    if result['test_evaluated']:
        for objective in ('self_scaled', 'fixed_reference'):
            rows = result['test_results'][objective]['per_seed']
            assert [row['seed'] for row in rows] == spec['model_seeds']
            for row in [*rows, result['test_results'][objective]['ensemble']]:
                summary_tools.validated_prediction_arrays(row, spec['cap'], str(path)+':'+objective)
    return {'result_sha256': sha(path), 'fit_count': len(spec['model_seeds']),
            'verified_checkpoint_count': checkpoint_count, 'test_evaluated': result['test_evaluated']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    options = parser.parse_args()
    plan = create_plan()
    if options.prepare_only:
        print(json.dumps({'plan_path': str(PLAN), 'jobs': len(plan['jobs']),
                          'planned_test_fits': sum(len(spec['model_seeds']) for spec in plan['jobs'].values() if spec['stage'] == 'five_seed_test'),
                          'planned_guarded_carrier_fits': 10}, indent=2))
        return
    lock = (OUTPUT/'worker.lock').open('a+b')
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'1'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    for group in ('source_sha256', 'selection_input_sha256', 'raw_data_sha256'):
        for path, digest in plan[group].items():
            assert sha(Path(path)) == digest, path
    assert sha(BIB) == BIB_SHA
    state = {'run_id': plan['run_id'], 'worker_pid': os.getpid(), 'started_utc': utc(),
             'status': 'running', 'jobs': {}, 'manuscript_modified': False, 'bib_modified': False}
    state_lock = threading.Lock()
    def save():
        with state_lock:
            state['updated_utc'] = utc()
            state['completed_test_fits'] = sum(row.get('fit_count', 0) for key, row in state['jobs'].items() if plan['jobs'][key]['stage'] == 'five_seed_test' and row['status'] == 'completed')
            state['completed_guarded_carrier_fits'] = sum(row.get('fit_count', 0) for key, row in state['jobs'].items() if plan['jobs'][key]['stage'] == 'guarded_carrier' and row['status'] == 'completed')
            atomic(STATUS, state)
    def record(key, value, replace=False):
        with state_lock:
            if replace:
                state['jobs'][key] = value
            else:
                state['jobs'][key].update(value)
        save()
    def execute(key):
        spec = plan['jobs'][key]
        path = Path(spec['result_path'])
        if path.exists():
            record(key, {'status': 'completed', **validate(path, spec, plan)}, replace=True)
            return
        logdir = OUTPUT/'logs'; logdir.mkdir(exist_ok=True)
        prior = list(logdir.glob(key+'_attempt*.log'))
        if prior:
            raise RuntimeError(f'Incomplete prior job requires explicit audit before resumption: {key}')
        log = logdir/(key+'_attempt1.log')
        command = [sys.executable, '-u', str(ROOT/'runtime_capture_trainer.py'),
                   '--reproduction-plan', str(PLAN), '--spec-key', key, *spec['native_cli_arguments']]
        record(key, {'status': 'running', 'started_utc': utc(), 'log': str(log), 'command': command}, replace=True)
        with log.open('x', encoding='utf-8') as handle:
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            record(key, {'pid': process.pid})
            code = process.wait()
        record(key, {'finished_utc': utc(), 'returncode': code})
        if code != 0:
            record(key, {'status': 'failed'})
            raise RuntimeError(f'Job failed: {key}; inspect {log}')
        record(key, {'status': 'completed', **validate(path, spec, plan)})
    try:
        # Validate the diagnosed failing configuration beyond epoch 4 first.
        state['active_stage'] = 'guarded_carrier_failure_case'; save()
        execute('guard_FD002_cap125')
        state['active_stage'] = 'five_seed_test_replay'; save()
        keys = [key for key, spec in plan['jobs'].items() if spec['stage'] == 'five_seed_test']
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending = {executor.submit(execute, key) for key in keys}
            while pending:
                done, pending = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                save()
        state['active_stage'] = 'remaining_guarded_carrier_profile'; save()
        for key, spec in plan['jobs'].items():
            if spec['stage'] == 'guarded_carrier' and key != 'guard_FD002_cap125':
                execute(key)
        state['active_stage'] = 'summarize_and_export'; save()
        process = subprocess.run([sys.executable, str(ROOT/'summarize_new_test_replay.py')], cwd=ROOT)
        if process.returncode != 0:
            raise RuntimeError('Independent final export failed')
        state['active_stage'] = None; state['status'] = 'complete'; state['finished_utc'] = utc(); save()
        atomic(OUTPUT/'COMPLETE.json', {'run_id': plan['run_id'], 'plan_sha256': sha(PLAN),
                                      'finished_utc': utc(), 'test_fits': state['completed_test_fits'],
                                      'guarded_carrier_fits': state['completed_guarded_carrier_fits']})
    except Exception as error:
        state['status'] = 'failed'; state['error'] = str(error); state['finished_utc'] = utc(); save()
        raise


if __name__ == '__main__':
    main()
