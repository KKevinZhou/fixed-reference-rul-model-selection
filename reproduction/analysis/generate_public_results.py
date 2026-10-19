"""Render new manuscript assets from the complete, auditable local rerun.

Use the original pure scientific summarizers/formatters. Do not invoke or
imitate the historical official-attestation entry point, and never edit Bib.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent/'reproduction_attempt_20261010'
sys.path.insert(0, str(REPRO))
import continue_reproduction_20261011 as run
import generate_manuscript_assets as assets

MANUSCRIPT = HERE/'staging_manuscript'
SUMMARY = HERE/'verified_summary'
METHODS = {'cap_self_scaled': 'cap_grid_self_scaled', 'cap_fixed_reference': 'cap_grid_fixed_reference',
           'cap_fixed_cap_125': 'fixed_cap_125'}


def csv(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(SUMMARY/name, index=False, encoding='utf-8-sig')
    return frame


def final_frames():
    plan = run.read(run.PLAN)
    lock = run.read(run.CAMPAIGN/'final_cap_decisions.json')
    selected = pd.read_csv(run.OUTPUT/'verified_exports'/'test_selected_methods.csv').set_index(['dataset', 'method'])
    finals = []
    for ds in run.protocol.DATASETS:
        for family in ('cap', 'tpe'):
            if family == 'tpe' and ds not in ('FD001', 'FD002'):
                continue
            for objective in ('self_scaled', 'fixed_reference', 'fixed_cap_125'):
                method = family+'_'+objective
                key = f'cap_{ds}_{int(lock["decisions"][ds][objective]["selected_cap"])}' if family == 'cap' else f'tpe_{ds}_{objective}'
                spec = plan['jobs'][key]
                run.validate(Path(spec['result_path']), spec, plan)
                result = run.read(Path(spec['result_path']))
                value = result['test_results']['self_scaled' if objective == 'self_scaled' else 'fixed_reference']
                cap = spec['cap']
                units, pc, pr, y = run.summary_tools.validated_prediction_arrays(value['ensemble'], cap, key)
                tc, tr = np.minimum(y, cap), np.minimum(y, 125)
                run.verify_test_truth(ds, y)
                assert units == tuple(str(i) for i in range(1, len(y)+1))
                assert np.allclose(value['ensemble']['raw_predictions'], np.median([r['raw_predictions'] for r in value['per_seed']], axis=0), atol=1e-10, rtol=1e-10)
                stats = selected.loc[(ds, method)]
                metrics = np.asarray([r['reference_rmse'] for r in value['per_seed']])
                assert np.isclose(metrics.mean(), stats.seed_rmse_mean, atol=1e-10, rtol=1e-10)
                own = run.summary_tools.rmse(pc, tc)
                ref = run.summary_tools.rmse(pr, tr)
                assert np.isclose(ref, stats.ensemble_rmse, atol=1e-10, rtol=1e-10)
                own_mse=run.summary_tools.mse(pc,tc)
                original_gap = run.summary_tools.mse(pc, y)-own_mse
                reference_gap = run.summary_tools.mse(pc, tr)-own_mse
                e2 = float(np.mean(np.maximum(y-cap, 0)**2))
                er2 = float(np.mean(np.maximum(tr-cap, 0)**2))
                assert original_gap >= e2-1e-7*(1+abs(original_gap))
                assert reference_gap >= er2-1e-7*(1+abs(reference_gap))
                row = {'dataset': ds, 'method': METHODS.get(method, method), 'train_cap': cap,
                       'reference_rmse_mean': stats.seed_rmse_mean, 'reference_rmse_std': stats.seed_rmse_std,
                       'reference_rmse_ensemble': ref, 'own_rmse_ensemble': own,
                       'candidate_target_common_support_rmse_ensemble': run.summary_tools.rmse(pr, tc),
                       'candidate_support_reference_rmse_ensemble': run.summary_tools.rmse(pc, tr),
                       'candidate_support_original_rmse_ensemble': run.summary_tools.rmse(pc, y),
                       'original_rmse_ensemble': run.summary_tools.rmse(pr, y),
                       'reference_rmse_bootstrap_low': stats.engine_bootstrap_low, 'reference_rmse_bootstrap_high': stats.engine_bootstrap_high,
                       'nasa_score_reference_ensemble': run.summary_tools.nasa_score(pr, tr),
                       'saturation_fraction': float(np.mean(y>cap)), 'exposure_bound': np.sqrt(e2),
                       'reference_exposure_bound': np.sqrt(er2),
                       'candidate_hidden_mse_gap': original_gap, 'exposure_bound_squared': e2,
                       'reference_hidden_mse_gap': reference_gap, 'reference_exposure_bound_squared': er2,
                       'bound_fraction_of_hidden_gap': e2/original_gap if original_gap>0 else None,
                       'reference_bound_fraction_of_hidden_gap': er2/reference_gap if reference_gap>0 else None,
                       'n_test_engines': len(y), 'result_sha256': run.sha(Path(spec['result_path']))}
                decision = run.summary_tools.reference_decision_metrics(pr, tr)
                row.update(mean_error_ensemble=decision.pop('mean_error'))
                row.update(decision)
                finals.append(row)
    pairs = pd.read_csv(run.OUTPUT/'verified_exports'/'test_paired_contrasts.csv').rename(columns={
        'paired_seed_differences': 'paired_difference_values', 'negative_seed_count': 'left_better_seed_count',
        'nonzero_pairs': 'wilcoxon_nonzero_pairs', 'ensemble_difference': 'paired_engine_rmse_difference',
        'paired_engine_ci_low': 'paired_engine_bootstrap_95ci_low', 'paired_engine_ci_high': 'paired_engine_bootstrap_95ci_high',
        'holm_all12_pvalue': 'wilcoxon_holm_all12_pvalue', 'holm_primary4_pvalue': 'wilcoxon_holm_primary_pvalue',
        'holm_secondary8_pvalue': 'wilcoxon_holm_secondary_pvalue'})
    pairs['n_model_seeds'] = 5
    pairs['paired_model_seeds'] = json.dumps([45,46,47,48,49])
    pairs['wilcoxon_method'] = 'exact_sign_flip_average_ranks_zero_dropped'
    pairs['n_test_engines'] = pairs.dataset.map({ds: int(next(r['n_test_engines'] for r in finals if r['dataset']==ds)) for ds in run.protocol.DATASETS})
    return pd.DataFrame(finals), pairs


def carrier_tables():
    lines = ['% New local execution: original precision and guarded sensitivity kept separate.',
             r'\begin{table}[htbp]', r'\centering\small',
             r'\caption{Original-precision dual-stream validation profile (split 20260724, seed 42). Own-target and fixed-reference columns use different target scales. The failed cell is excluded from cap selection.}',
             r'\label{sup:tab:carrier-transfer}', r'\begin{tabular}{llrrr}', r'\toprule',
             r'Subset & Status & Cap & Own val. & Ref. val. \\', r'\midrule']
    rows = []
    for ds in ('FD001','FD002'):
        for cap in (90,100,110,115,125):
            paths = list((run.CAMPAIGN/'carrier_transfer'/'results').glob(f'{ds}_dualstream_transfer_cap{cap}_*.json'))
            if not paths:
                assert ds=='FD002' and cap==125
                lines.append(f'{ds} & Non-finite loss & {cap} & -- & -- '+r'\\'); continue
            assert len(paths)==1, paths
            result = run.read(paths[0])
            own = result['validation_summary']['self_scaled']['own_rmse']['mean']
            ref = result['validation_summary']['fixed_reference']['reference_rmse']['mean']
            rows.append({'dataset':ds,'cap':cap,'own_rmse':own,'reference_rmse':ref})
            lines.append(f'{ds} & Completed & {cap} & {own:.2f} & {ref:.2f} '+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\end{table}','',r'\begin{table}[htbp]',r'\centering\small',
              r'\caption{Separate sensitivity rerun with FP32 multihead attention and original AMP elsewhere. All ten cells share this numerical policy; none replaces an original-precision failure. Own and reference columns use distinct target scales.}',
              r'\label{sup:tab:guarded-carrier}',r'\begin{tabular}{lrrr}',r'\toprule',r'Subset & Cap & Own val. & Ref. val. \\',r'\midrule']
    guarded = pd.read_csv(run.OUTPUT/'verified_exports'/'guarded_carrier_profile.csv').sort_values(['dataset','cap'])
    for _,row in guarded.iterrows():
        lines.append(f'{row.dataset} & {int(row.cap)} & {row.self_scaled_own_rmse:.2f} & {row.fixed_reference_rmse:.2f} '+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\end{table}','']
    (MANUSCRIPT/'generated_carrier_transfer_table.tex').write_text('\n'.join(lines),encoding='utf-8')
    csv('original_precision_carrier.csv',rows)


def test_figure(finals,destination):
    """Keep labels inside the figure and place the legend above the bars."""
    import matplotlib.pyplot as plt
    figure,axis=plt.subplots(figsize=(7.2,3.6))
    x=np.arange(4); width=0.24
    for offset,method in enumerate(assets.METHODS):
        rows=finals[finals.method==method].set_index('dataset').loc[assets.DATASETS]
        axis.bar(x+(offset-1)*width,rows.reference_rmse_mean,width,
                 yerr=rows.reference_rmse_std,color=assets.COLORS[method],
                 hatch=assets.HATCHES[method],edgecolor='#222222',linewidth=.5,
                 capsize=2.5,label=assets.LABELS[method])
    axis.set_xticks(x,assets.DATASETS)
    axis.set_ylabel('Test RMSE (cycles)')
    axis.grid(axis='y',alpha=.2)
    figure.legend(*axis.get_legend_handles_labels(),frameon=False,ncol=3,
                  loc='upper center',bbox_to_anchor=(.52,1.0))
    figure.tight_layout(rect=(0,0,1,.9))
    figure.savefig(destination,bbox_inches='tight')
    plt.close(figure)


def main():
    assert (run.OUTPUT/'COMPLETE.json').exists()
    assert (REPRO/'author_reported_cap_replay_20261011'/'COMPLETE.json').exists()
    assert run.sha(run.BIB)==run.BIB_SHA and run.sha(MANUSCRIPT/'reference.bib')==run.BIB_SHA
    for group in ('source_sha256','selection_input_sha256'):
        for path,digest in run.read(run.PLAN)[group].items():
            assert run.sha(Path(path))==digest,path
    SUMMARY.mkdir(exist_ok=True)
    s = run.summary_tools
    validation,bounds=s.screening_rows(run.CAMPAIGN)
    stability=s.stability_rows(run.CAMPAIGN)
    cap=s.cap_summary(validation,stability)
    aggregation=s.aggregation_sensitivity(validation,stability)
    decomposition=s.decomposition_summary(validation,stability)
    decisions=s.decision_rows(run.CAMPAIGN,cap)
    tpe=s.tpe_rows(run.CAMPAIGN)
    robustness=s.tpe_robustness_rows(run.CAMPAIGN,tpe)
    finals,pairs=final_frames()
    frames={name:csv(name,rows) for name,rows in [
        ('validation_long.csv',validation),('exposure_bound_audit.csv',bounds),('stability_long.csv',stability),
        ('cap_summary.csv',cap),('aggregation_sensitivity.csv',aggregation),('target_support_decomposition.csv',decomposition),
        ('locked_decisions.csv',decisions),('tpe_locks.csv',tpe),('tpe_robustness.csv',robustness),
        ('final_test.csv',finals),('paired_tests.csv',pairs)]}
    expected={'validation_long.csv':120,'exposure_bound_audit.csv':120,'stability_long.csv':80,
        'cap_summary.csv':40,'aggregation_sensitivity.csv':32,'target_support_decomposition.csv':160,
        'locked_decisions.csv':12,'tpe_locks.csv':6,'tpe_robustness.csv':6,'final_test.csv':18,'paired_tests.csv':12}
    assert {name:len(df) for name,df in frames.items()}==expected
    assert all(row['bound_satisfied'] for row in bounds)
    assets.validate_reportable_metrics(finals,pairs,frames['validation_long.csv'])
    figures=MANUSCRIPT/'figures'
    assets.validation_figure(frames['cap_summary.csv'],figures/'fig_confirmatory_validation.pdf')
    test_figure(finals,figures/'fig_confirmatory_test_rmse.pdf')
    figure_values=finals[finals.method.isin(assets.METHODS)][['dataset','method','reference_rmse_mean','reference_rmse_std']].copy()
    figure_values['method']=figure_values.method.map(assets.LABELS)
    figure_values.columns=['subset','selection','mean_rmse','sample_sd']
    figure_values.to_csv(figures/'fig_confirmatory_test_rmse_values.csv',index=False,encoding='utf-8-sig')
    provenance='% Verified new local execution, 2026-10-10--11; generated 2026-10-13.\n% Historical campaign attestation is not claimed.\n'
    primary=assets.latex_tables(frames['locked_decisions.csv'],frames['tpe_locks.csv'],finals,pairs)
    primary=primary.replace('Caps locked before any standard test-set outcome evaluation.','Caps locked before the current five-seed test phase; prior standard-test contact is disclosed.')
    (MANUSCRIPT/'generated_confirmatory_tables.tex').write_text(provenance+primary,encoding='utf-8')
    supplement=''.join([
        assets.supplementary_pair_table(pairs),assets.supplementary_individual_table(finals),
        assets.supplementary_tpe_robustness_table(frames['tpe_robustness.csv']),
        assets.supplementary_aggregation_sensitivity_table(frames['aggregation_sensitivity.csv']),
        assets.supplementary_validation_two_by_two_table(frames['target_support_decomposition.csv']),
        assets.supplementary_checkpoint_pool_selection_table(frames['target_support_decomposition.csv']),
        assets.supplementary_validation_saturation_table(frames['validation_long.csv']),
        assets.supplementary_two_by_two_contrast_table(finals),assets.supplementary_maintenance_threshold_table(finals),
        assets.supplementary_saturation_table(finals)])
    (MANUSCRIPT/'generated_supplementary_tables.tex').write_text(provenance+supplement,encoding='utf-8')
    a,n,c,h,g=assets.result_text_fragments(frames['locked_decisions.csv'],frames['tpe_locks.csv'],pairs,finals,frames['target_support_decomposition.csv'])
    n=n.replace('This diagnostic localizes the observed cap-ranking change within the archived candidates;',
        'The sixteen contrasts in each category reuse eight checkpoint pools and are not independent experiments. This diagnostic localizes the observed cap-ranking change within the archived candidates;')
    for name,text in [('generated_abstract_result.tex',a),('generated_results_narrative.tex',n),('generated_graphical_result.tex',g)]:
        (MANUSCRIPT/name).write_text(provenance+text,encoding='utf-8')
    carrier_tables()
    pd.DataFrame(assets.checkpoint_pool_selection_rows(frames['target_support_decomposition.csv'])).to_csv(SUMMARY/'checkpoint_pool_reranking.csv',index=False,encoding='utf-8-sig')
    receipt={'protocol':'new-local-manuscript-summary-v1','data_source':str(REPRO),'created_utc':run.utc(),
        'historical_attestation_claimed':False,'all_primary_methods_retained':True,'selection_from_test_outcomes':False,
        'summary_counts':expected,'new_execution_plan_sha256':run.sha(run.PLAN),'bib_unchanged':True,
        'csv_sha256':{p.name:run.sha(p) for p in SUMMARY.glob('*.csv')},
        'formatter_sha256':run.sha(REPRO/'source'/'generate_manuscript_assets.py'),'generator_sha256':run.sha(Path(__file__))}
    receipt['figure_values_sha256']=run.sha(figures/'fig_confirmatory_test_rmse_values.csv')
    run.atomic(SUMMARY/'NEW_MANUSCRIPT_RESULTS_VERIFIED.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
