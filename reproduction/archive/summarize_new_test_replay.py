"""Independent exports for the new test execution and guarded carrier profile."""
from __future__ import annotations
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import continue_reproduction_20261011 as runner

ROOT = Path(__file__).resolve().parent
tools = runner.summary_tools


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def near(left, right):
    if not np.isclose(left, right, rtol=1e-8, atol=1e-8):
        raise RuntimeError(f'Stored metric mismatch: {left} != {right}')


def main():
    plan = runner.read(runner.PLAN)
    loaded, raw_rows, metric_rows, carrier_rows = {}, [], [], []
    weights = 0
    for key, spec in plan['jobs'].items():
        path = Path(spec['result_path'])
        checked = runner.validate(path, spec, plan)
        weights += checked['verified_checkpoint_count']
        result = runner.read(path)
        loaded[key] = result
        if spec['stage'] == 'guarded_carrier':
            carrier_rows.append({'dataset': result['dataset'], 'cap': result['train_cap'],
                                 'self_scaled_own_rmse': result['validation_summary']['self_scaled']['own_rmse']['mean'],
                                 'fixed_reference_rmse': result['validation_summary']['fixed_reference']['reference_rmse']['mean'],
                                 'numerical_policy': spec['numerical_policy'], 'result_sha256': checked['result_sha256']})
            continue
        truth = np.loadtxt(runner.DATA/'Experiment_FD001'/'CMAPSSData'/f"RUL_{result['dataset']}.txt").reshape(-1)
        for objective in ('self_scaled', 'fixed_reference'):
            value = result['test_results'][objective]
            for seed_row in [*value['per_seed'], {'seed': 'median_ensemble', **value['ensemble']}]:
                units, candidate, common, original = tools.validated_prediction_arrays(seed_row, spec['cap'], key)
                assert units == tuple(str(index) for index in range(1, len(truth)+1))
                assert np.array_equal(original, truth)
                own = tools.rmse(candidate, np.minimum(original, spec['cap']))
                reference = tools.rmse(common, np.minimum(original, 125))
                uncapped = tools.rmse(common, original)
                nasa = tools.nasa_score(common, np.minimum(original, 125))
                for name, metric in [('own_rmse', own), ('reference_rmse', reference),
                                     ('original_rmse', uncapped), ('nasa_score_reference', nasa)]:
                    near(seed_row[name], metric)
                base = {'job': key, 'dataset': result['dataset'], 'cap': spec['cap'],
                        'objective': objective, 'seed': seed_row['seed']}
                metric_rows.append({**base, 'own_rmse': own, 'reference_rmse': reference,
                                    'uncapped_rmse': uncapped, 'nasa_reference': nasa,
                                    'test_engines': len(units), 'result_sha256': checked['result_sha256']})
                for unit, raw, cand, pred, target in zip(units, seed_row['raw_predictions'], candidate, common, original, strict=True):
                    raw_rows.append({**base, 'unit': unit, 'raw_prediction': raw,
                                     'candidate_prediction': float(cand), 'common_prediction': float(pred),
                                     'original_rul': float(target)})
            ensemble = value['ensemble']
            expected = np.median([row['raw_predictions'] for row in value['per_seed']], axis=0)
            assert np.allclose(expected, ensemble['raw_predictions'], rtol=1e-10, atol=1e-10)
    cap_lock = runner.read(runner.CAMPAIGN/'final_cap_decisions.json')
    methods = {}
    for dataset in runner.protocol.DATASETS:
        decisions = cap_lock['decisions'][dataset]
        methods[dataset] = {}
        for name in ('self_scaled', 'fixed_reference', 'fixed_cap_125'):
            cap = decisions[name]['selected_cap']
            objective = 'self_scaled' if name == 'self_scaled' else 'fixed_reference'
            methods[dataset]['cap_'+name] = loaded[f'cap_{dataset}_{cap}']['test_results'][objective]
        if dataset in ('FD001', 'FD002'):
            for name in ('self_scaled', 'fixed_reference', 'fixed_cap_125'):
                objective = 'self_scaled' if name == 'self_scaled' else 'fixed_reference'
                methods[dataset]['tpe_'+name] = loaded[f'tpe_{dataset}_{name}']['test_results'][objective]
    method_rows, contrasts = [], []
    for dataset, entries in methods.items():
        for name, value in entries.items():
            metrics = np.asarray([row['reference_rmse'] for row in value['per_seed']])
            ensemble = value['ensemble']
            target = np.minimum(ensemble['original_targets'], 125)
            low, high = tools.single_engine_bootstrap_rmse(ensemble['common_support_predictions'], target, f'{dataset}:{name}', resamples=2000)
            method_rows.append({'dataset': dataset, 'method': name,
                                'seed_rmse_mean': float(metrics.mean()), 'seed_rmse_std': float(metrics.std(ddof=1)),
                                'seed_rmse_values': json.dumps(metrics.tolist()),
                                'ensemble_rmse': ensemble['reference_rmse'],
                                'engine_bootstrap_low': low, 'engine_bootstrap_high': high})
        pairs = [('fixed_vs_self_cap_grid', 'cap_fixed_reference', 'cap_self_scaled'),
                 ('fixed_cap_grid_vs_fixed125', 'cap_fixed_reference', 'cap_fixed_cap_125')]
        if dataset in ('FD001', 'FD002'):
            pairs += [('fixed_vs_self_tpe', 'tpe_fixed_reference', 'tpe_self_scaled'),
                      ('fixed_reference_tpe_vs_fixed_cap_125_tpe', 'tpe_fixed_reference', 'tpe_fixed_cap_125')]
        for label, left_name, right_name in pairs:
            left, right = entries[left_name], entries[right_name]
            assert [row['seed'] for row in left['per_seed']] == [45, 46, 47, 48, 49]
            assert [row['seed'] for row in right['per_seed']] == [45, 46, 47, 48, 49]
            differences = np.asarray([l['reference_rmse']-r['reference_rmse'] for l, r in zip(left['per_seed'], right['per_seed'], strict=True)])
            statistic, pvalue, count = tools.exact_signed_rank_test(differences)
            le, re = left['ensemble'], right['ensemble']
            assert le['units'] == re['units'] and le['original_targets'] == re['original_targets']
            point, low, high, negative = tools.paired_engine_bootstrap(le['common_support_predictions'], re['common_support_predictions'], np.minimum(le['original_targets'], 125), f'{dataset}:{label}', resamples=10000)
            contrasts.append({'dataset': dataset, 'comparison': label,
                              'paired_seed_differences': json.dumps(differences.tolist()),
                              'mean_difference': float(differences.mean()), 'std_difference': float(differences.std(ddof=1)),
                              'median_difference': float(np.median(differences)), 'negative_seed_count': int((differences<0).sum()),
                              'wilcoxon_statistic': statistic, 'wilcoxon_pvalue': pvalue, 'nonzero_pairs': count,
                              'ensemble_difference': point, 'paired_engine_ci_low': low,
                              'paired_engine_ci_high': high, 'bootstrap_fraction_negative': negative})
    assert len(contrasts) == 12
    tools.add_holm_adjustment(contrasts, 'holm_all12_pvalue')
    tools.add_holm_adjustment([row for row in contrasts if row['comparison'] == 'fixed_vs_self_cap_grid'], 'holm_primary4_pvalue')
    tools.add_holm_adjustment([row for row in contrasts if row['comparison'] != 'fixed_vs_self_cap_grid'], 'holm_secondary8_pvalue')
    selections = {}
    for dataset in ('FD001', 'FD002'):
        profile = [row for row in carrier_rows if row['dataset'] == dataset]
        assert len(profile) == 5
        selections[dataset] = {name: int(min(profile, key=lambda row: (row[field], -row['cap']))['cap'])
                               for name, field in [('self_scaled', 'self_scaled_own_rmse'), ('fixed_reference', 'fixed_reference_rmse')]}
    for group in ('source_sha256', 'selection_input_sha256', 'raw_data_sha256'):
        for path, digest in plan[group].items():
            assert runner.sha(Path(path)) == digest, path
    assert runner.sha(runner.BIB) == runner.BIB_SHA
    out = runner.OUTPUT/'verified_exports'; out.mkdir(exist_ok=True)
    for name, rows in [('test_raw_predictions.csv', raw_rows), ('test_seed_metrics.csv', metric_rows),
                       ('test_selected_methods.csv', method_rows), ('test_paired_contrasts.csv', contrasts),
                       ('guarded_carrier_profile.csv', carrier_rows)]:
        write_csv(out/name, rows)
    report = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': runner.sha(runner.PLAN),
              'new_test_fits': 90, 'guarded_carrier_fits': 10, 'verified_new_checkpoints': weights,
              'raw_prediction_rows_including_ensembles': len(raw_rows), 'paired_contrasts': 12,
              'guarded_carrier_selections': selections, 'manuscript_modified': False, 'bib_modified': False,
              'legacy_original_full_qa_success': False, 'legacy_official_attestation': False,
              'interpretation': 'Actual new outputs; conditional descriptive reproduction with prior test contact, not recovery of historical artifacts'}
    runner.atomic(out/'FINAL_NEW_EXECUTION_VERIFIED.json', report)
    text = '# 本次继续复现结果\n\n这是本机的新执行，主模型保持原始精度；第二模型的注意力 FP32 修复另列。原论文精确数值仍须逐项比较，不能据此补造历史记录。\n\n'
    text += '| 子集 | 方法 | 五种子 RMSE 均值 ± 标准差 | 中位数集成 RMSE |\n|---|---|---:|---:|\n'
    text += '\n'.join(f"| {row['dataset']} | {row['method']} | {row['seed_rmse_mean']:.4f} ± {row['seed_rmse_std']:.4f} | {row['ensemble_rmse']:.4f} |" for row in method_rows)
    text += '\n\n已保存每个种子的逐发动机预测、模型权重、12 个配对差值与 Holm 校正，以及 10,000 次配对发动机 bootstrap。原始第二模型失败仍保留，修复版不混合计入原始精度结果。论文和 Bib 未改动。\n'
    (runner.OUTPUT/'RESULTS_CN.md').write_text(text, encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
