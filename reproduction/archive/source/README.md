# Fixed-Reference Scoring under Candidate-Dependent RUL Capping

Reproduction artifact for **"Fixed-Reference Scoring for Task-Consistent RUL
Model Selection under Tunable Target Capping."**

The artifact studies a failure mode in remaining-useful-life (RUL) model
selection. If each cap candidate `c` is scored against its own target
`T_c(y) = min(y, c)`, the optimizer changes the quantity being minimized.
The fixed-reference protocol keeps the training cap tunable while holding the
validation target and reporting transformation fixed.

> **Evidence boundary.** The authors inspected outcomes from the same standard
> C-MAPSS test split in exploratory work before this hash-bound rerun was
> specified. The controls below establish decision isolation only within the
> reported rerun; they do not constitute external preregistration or restore
> an unseen test set. The retained 20-run exploratory cap sweep analysed in
> the article is separated under `archive/exploratory_v1/`; other historical
> development outputs in the working directory are not distributed. All
> standard-test summaries are described as conditional, descriptive evidence.

## Auditable workflow and state machine

The standard C-MAPSS test and truth values remain unparsed and unevaluated
until all validation and HPO decisions are frozen. A post-lock,
pre-evaluation step reads file bytes only to verify SHA-256 fingerprints
against the independently archived NASA manifest. That manifest also anchors
the expected number of test units and a SHA-256 digest of the ordered
unit/target payload; the formal rerun does not reconstruct that payload until
the test-start marker exists. Run the stages in this order:

```text
three-split cap screen
  -> primary-split model-seed stability
  -> final cap lock
  -> equal-budget TPE studies
  -> frozen-winner split/seed sensitivity checks
  -> validation-only second-carrier sensitivity check
  -> selection-attempt manifest + full QA receipt
  -> pre-marker campaign inventory
  -> content-addressed test plan
  -> plan-bound standard test-set evaluation phase
  -> generated summaries
```

The scripts and completion artifacts are:

| Stage | Script | Required completion artifact |
|---|---|---|
| Cap screen | `run_confirmatory_cap_sweep.py` | `cap_sweep_summary.json` and `locked_cap_decisions.json` |
| Seed stability/final cap lock | `run_confirmatory_seed_stability.py` | `stability_summary.json` and `final_cap_decisions.json` |
| Equal-budget HPO | `run_confirmatory_tpe.py` or `wait_and_run_tpe_suite.py` | six files under `tpe_locks/` and `TPE_SUITE_COMPLETE` |
| Frozen-winner split/seed sensitivity | `run_tpe_robustness.py` | `tpe_robustness_summary.json` |
| Second-carrier sensitivity | `run_carrier_transfer.py` | `carrier_transfer_plan.json` and `carrier_transfer_summary.json` |
| Exact second-carrier failure recovery | `retry_carrier_transfer_failures.py` | `carrier_transfer_retry_manifest.json` and `carrier_transfer_retry_summary.json` |
| Selection-attempt provenance | `create_selection_attempt_manifest.py` | `SELECTION_ATTEMPT_MANIFEST.json` |
| Full artifact QA | `run_qa_suite.py` | `qa/pytest-junit.xml` and `qa/pytest_run_manifest.json` |
| Pre-marker inventory | `preflight_confirmatory.py` | `PREMARKER_CAMPAIGN_INVENTORY.json` and `preflight_report.json` |
| Standard test-set descriptive evaluation | `run_confirmatory_final.py` | `immutable_test_plan.json`, `TEST_EVALUATION_STARTED.json`, `TEST_DATA_VERIFIED.json`, plan-bound files under `final_receipts/`, `final_test_summary.json`, and `TEST_EVALUATED` |
| Tables/audit | `summarize_confirmatory.py` | files under `summaries/` |
| Manuscript assets | `generate_manuscript_assets.py` | locked LaTeX tables and two PDF figures generated from the summary CSV files |

The separate source-level audit of the published RULe workflow is reproducible
without executing or redistributing that third-party code:

```bash
python audit_public_workflow.py --receipt public_workflow_audit.json
```

This command fetches four text files from commit
`4cf4563d1abcf350de9958a0de26d2d0126c819d`, verifies their pinned SHA-256
digests, and checks the exact line fragments cited by the Supplementary
Material. It does not open C-MAPSS test or truth data.

`wait_and_finalize_confirmatory.py` is an optional fail-closed orchestrator for
the last three rows. It waits for every validation/HPO prerequisite, runs the
strict preflight and single standard test-set wrapper, strictly resumes only the
same immutable plan after interruption, then creates summaries and manuscript
assets. It also waits for the validation-only second-carrier sensitivity check,
which is isolated from the primary decision set. It never uploads or publishes
files. A stale watcher lock is retained
under a timestamped name before a replacement watcher starts.

`run_confirmatory_final.py` refuses to start unless the final cap lock, all six
TPE locks, and the frozen-config robustness summary agree with their hashed
manifests. If execution is interrupted before a completion ledger exists, the
same content-addressed test plan permits at most one exact-configuration
retry. A completed nonzero process exit is terminal and cannot be retried into
a successful cell; completed matching result files are validated and skipped.
Candidate choices cannot be changed after the plan is created.
The test plan contains the canonical C-MAPSS test/truth SHA-256 fingerprints,
the expected unit count and ordered unit/target digest, and the complete
expected configuration for every planned result. These anchors are frozen in
`canonical_cmapss_sha256.json`; all twelve local train/test/truth files must
match the independently archived NASA PCoE source. Plan construction copies
the trusted test/truth anchors without opening local test/truth files. The
wrapper then writes the immutable plan and test-start marker, verifies local
test/truth bytes, and records the plan-bound result in
`TEST_DATA_VERIFIED.json`. Before a training process starts, the wrapper writes
a create-only launch authorization with a unique nonce. The
`final_attested_trainer.py` shim calls the byte-frozen training module,
preserves the per-seed prediction arrays already computed by the evaluator,
and writes the nonce and plan/start/launch hashes into the result. A process-wide
lock prevents concurrent plan-bound test wrappers, and the completion audit requires
the result set to equal the planned set exactly. The outer validator
separately reconstructs all per-seed metrics, five-seed summaries, and
median-ensemble metrics from the archived raw arrays and rejects any unit/target
payload whose digest differs from the canonical anchor.
The decision-proxy fields emitted by a direct
`confirmatory_rul_train.py` invocation are not the formal reporting source:
the attested final runner and content-addressed summarizer reconstruct those
fields from archived arrays against the fixed target \(T_{125}\).
Every subprocess
launch also creates an immutable attempt-start record; exits create a matching
completion record with the log hash. If a hardware/process interruption
requires same-plan recovery, the incomplete and completed attempts are both
bound into the final receipt. After validation
the wrapper writes a create-only completion receipt binding the plan, start
marker, result, configuration, and test/truth fingerprints. Resumption accepts
a result only with this matching hash chain.

Files named `strict_*`, `run_strict_*`, and older result folders in the
development workspace are **exploratory-v1 provenance**. They are not used in
the audited manuscript claims because their preprocessing boundary and
validation task were less strict. The release archive keeps any diagnostic
outputs, when included, under `archive/exploratory_v1/` rather than beside the
formal campaign.

The formal campaign manifest fixes `carrier: compact_tcn`. That carrier is
defined directly in `confirmatory_rul_train.py`. The unchanged
`baselines/models_v3.py` file is retained only because the frozen trainer
imports it and the source-hash closure records it; its dual-stream and
classification classes are not instantiated by the reported campaign and are
not claimed as contributions of this article.

## Definitions and prediction-clipping ranges

For original RUL `y`, training cap `c`, reporting cap `r`, and raw prediction
`z`, the artifact stores three prediction/evaluation views:

- candidate support: `clip(z, 0, c)`;
- common reporting clip: `clip(z, 0, r)`;
- raw prediction: `z`.

It reports:

- self-scaled target: `min(y, c)`;
- fixed-reference target: `min(y, r)`;
- cap-affected fraction: `mean(y > c)`;
- uncapped exposure: `E(c) = sqrt(mean(max(y - c, 0)^2))`;
- reference-scale exposure for `c <= r`:
  `E_r(c) = sqrt(mean(max(min(y, r) - c, 0)^2))`.

The exposures can be computed from labelled run-to-failure targets without
fitting a model. Their interpretation as lower bounds on the aligned MSE gaps
is conditional on **candidate-ceiling predictions**. The theorem does not apply
unchanged to common-range or raw predictions. The summaries report the
complete target/output-ceiling 2x2 rescore: candidate or reference target crossed
with candidate or common prediction-clipping ceilings, together with the target-by-ceiling
interaction on the MSE scale. These are descriptive rescoring contrasts, not
causal or additive performance attributions. Raw predictions are archived for
audit, but all horizontal final comparisons use the reference target and
common reporting clip.

## Environment

Python 3.11 or newer is required by the pinned SciPy release; the reference
campaign used Python 3.12.3:

```bash
python -m pip install -r requirements-lock.txt
python -m pytest tests -q
```

Every result records Python, PyTorch, CUDA, NumPy, pandas, scikit-learn, cuDNN,
device, source, preprocessing, checkpoint, and raw-data fingerprints. CPU
execution is supported but substantially slower. Runtime depends on hardware;
progress is visible in `logs/`, `stability_logs/`, `tpe_logs/`,
`tpe_robustness_logs/`, `carrier_transfer/logs/`, and `final_logs/`.
`requirements-lock.txt` pins the direct and transitive Python packages used by
the reference environment (without claiming wheel-level supply-chain hashes),
`environment-lock.txt` records hardware/runtime provenance, and
`requirements.txt` gives supported version ranges. CUDA kernels are requested
in deterministic warning mode; the artifact does not claim bitwise identity
across devices, drivers, or library versions.

The planned workload is explicit: 60 one-seed cap-screen jobs, 20
two-seed stability jobs, 72 one-seed TPE trials, 18 frozen-winner robustness
results covering 24 model-seed evaluations, 10 planned validation-only
second-carrier cells (eight completed fits and two terminal repeated
non-finite-loss failures in the archived campaign), and 18 immutable-plan
standard-test configurations with five seeds each. Each stage is
resumable, and every result records elapsed time so users can estimate remaining
runtime on their own system.

The optional `wait_and_run_seed_stability_parallel.py` wrapper executes only
independent dataset/cap stability cells concurrently (at most three workers).
It leaves the frozen trainer, seeds, splits, budgets, and canonical decision
driver unchanged; the canonical driver revalidates every cached cell before
creating the lock.

Likewise, `resume_cap_sweep_parallel.py` can resume missing independent
screening cells with at most three workers. It first checks the immutable
campaign manifest and finally invokes the canonical cap-sweep driver, which
revalidates all 60 cells before creating the screening lock.

## Data layout

Download the exact NASA archive bound by
`canonical_cmapss_sha256.json`:

```text
https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
outer SHA-256: c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2
nested CMAPSSData.zip SHA-256: 74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f
```

Raw datasets are not redistributed. Verify both archive hashes, extract the
nested archive, and place the files under:

```text
<data_root>/Experiment_FD001/CMAPSSData/
  train_FD001.txt ... train_FD004.txt
  test_FD001.txt  ... test_FD004.txt
  RUL_FD001.txt   ... RUL_FD004.txt
```

Validation-only stages open only `train_FD00x.txt`. After all validation/HPO
decisions are frozen, the content-addressed plan is built from the trusted
manifest and written without opening local test/truth files. Local test and
truth bytes are hashed only after `TEST_EVALUATION_STARTED.json` exists, and
their columns and target values are parsed only after the matching
`TEST_DATA_VERIFIED.json` receipt exists.

## Smoke test

```bash
python confirmatory_rul_train.py \
  --dataset FD001 \
  --campaign_id smoke \
  --tag smoke_cap125 \
  --train_cap 125 \
  --reporting_cap 125 \
  --split_seed 20260724 \
  --seeds 1 \
  --epochs 1 \
  --validation_only
```

Confirm that the JSON contains `"validation_only": true`,
`"test_evaluated": false`, no test fingerprint, disjoint training/validation
engine lists, and one endpoint record per validation engine.

## End-to-end audited commands

Run the three-split cap screen:

```bash
python run_confirmatory_cap_sweep.py \
  --data_root /path/to/data_root \
  --output_root /path/to/ress_confirmatory_v2
```

Add model seeds 43 and 44 on the primary split and create the final cap lock:

```bash
python run_confirmatory_seed_stability.py \
  --data_root /path/to/data_root \
  --output_root /path/to/ress_confirmatory_v2
```

Run 12-trial TPE studies for FD001/FD002 under self-scaled selection,
fixed-reference selection, and the fixed-cap-125 control:

```bash
python wait_and_run_tpe_suite.py \
  --data_root /path/to/data_root \
  --output_root /path/to/ress_confirmatory_v2 \
  --trials 12
```

Evaluate every frozen TPE winner on additional model seeds and validation
splits held out from TPE optimization:

```bash
python run_tpe_robustness.py \
  --data_root /path/to/data_root \
  --output_root /path/to/ress_confirmatory_v2
```

Before standard-test plan creation, repeat the primary validation cap profile
on FD001 and FD002 with the pre-existing dual-stream carrier:

```bash
python run_carrier_transfer.py \
  --data_root /path/to/data_root \
  --campaign_root /path/to/ress_confirmatory_v2
```

If a second-carrier cell terminates with an explicitly logged non-finite
training loss, first allow the original runner to stop and retain its log.
At most one sequential retry with the exact same frozen configuration and
seed can then be launched with:

```bash
python retry_carrier_transfer_failures.py \
  --data_root /path/to/data_root \
  --campaign_root /path/to/ress_confirmatory_v2
```

The recovery helper refuses unclassified failures and any existing
standard-test state. It changes no hyperparameter, preserves the original
failed log, creates a hash-bound retry manifest before execution, and uses a
separate retry log. If the exact retry fails again, the cell remains a
reported training failure; the helper does not tune around it. It writes a
terminal carrier summary that retains every valid cell, identifies every
missing cell, and suppresses cap selection for an incomplete subset. Because
this check is descriptive and cannot alter the primary locks, that audited
terminal state permits the independent standard-test campaign to continue.

Only after all preceding completion artifacts exist, evaluate the immutable
standard test-set descriptive evaluation plan:

```bash
python run_confirmatory_final.py \
  --data_root /path/to/data_root \
  --output_root /path/to/ress_confirmatory_v2 \
  --final_seeds 5
```

Generate all auditable CSV and LaTeX fragments:

```bash
python summarize_confirmatory.py \
  --data_root /path/to/data_root \
  --output_root /path/to/ress_confirmatory_v2
```

The summarizer fails if either the uncapped- or reference-scale
candidate-ceiling exposure-bound check fails.
It also revalidates the immutable plan/start/result/launch/receipt chain,
recomputes scalar metrics from archived arrays and per-seed values, removes
stale empty tables, and writes SHA-256-bound summary provenance. The manuscript
asset generator refuses partial or mixed summary directories.
Final comparisons include five-seed dispersion, paired engine-bootstrap
intervals, reference-scale NASA score, signed error, and
delayed/premature maintenance-action rates.

Generate the LaTeX fragments and vector figures from the hash-bound summaries:

```bash
python generate_manuscript_assets.py \
  --summary_dir /path/to/ress_confirmatory_v2/summaries \
  --manuscript_dir /path/to/manuscript
```

The archived software does not redistribute the journal manuscript. To
rebuild the standalone result fragments and vector figures into an empty
directory, add `--assets_only`; this mode does not require cover-letter or
Highlights templates.

Replace the placeholder paths with local paths. No script depends on a
particular drive letter when the command-line arguments are supplied.

For example, the same summary command on a portable layout is:

```bash
python summarize_confirmatory.py \
  --data_root /path/to/NASA_Lab \
  --output_root /path/to/ress_confirmatory_v2
```

After the final chain is complete, create the Zenodo draft and reserve its
DOI without publishing. Finalize the same DOI and intended publication date
across the CFF, manuscript, bibliography, and cover letter:

```bash
python prepare_release_metadata.py \
  --doi 10.5281/zenodo.RECORD_ID \
  --date-released YYYY-MM-DD
```

Only then build the deterministic Zenodo archive:

```bash
python build_release.py \
  --artifact_root /path/to/reproduction_artifact \
  --confirmatory_root /path/to/ress_confirmatory_v2 \
  --output /path/to/candidate-dependent-target-capping-artifact-v1.0.0.zip
```

The builder refuses missing DOI/date metadata or an incomplete or
hash-inconsistent final chain, packages
the exact validation evidence set admitted by preflight, retains only the
plan-bound attempt logs bound by the completion receipts, excludes other
transient logs and lock files, and writes `MANIFEST.sha256` plus
`RELEASE_PROVENANCE.json` inside the archive. The dependency-free verifier
reconstructs all plan-bound per-seed scalars, five-seed summaries, median
ensembles, canonical unit/target bindings, and attempt/log chains. It checks
the hashes of the validation, TPE, robustness, and strict-preflight evidence;
semantic reconstruction of Optuna studies is performed by
`preflight_confirmatory.py` in the full environment. Verify
either the ZIP or an extracted copy on any filesystem:

```bash
python verify_release.py /path/to/candidate-dependent-target-capping-artifact-v1.0.0.zip
python verify_release.py /path/to/extracted-release
```

## Manuscript asset map

`summarize_confirmatory.py` creates the source CSV files and
`generate_manuscript_assets.py` consumes only their hash-bound complete set:

- validation cap profiles and locked-cap table:
  `cap_summary.csv`, `locked_decisions.csv`, and
  `fig_confirmatory_validation.pdf`;
- equal-trial search and frozen-winner checks:
  `tpe_locks.csv` and `tpe_robustness.csv`;
- plan-bound standard-test common-target results and paired comparisons:
  `final_test.csv`, `paired_tests.csv`, and
  `fig_confirmatory_test_rmse.pdf`;
- supplementary target/output-ceiling, checkpoint-pool cap-reranking,
  aggregation, target-capping, and maintenance diagnostics:
  `target_support_decomposition.csv`, `aggregation_sensitivity.csv`, and the
  remaining summary CSV files.

## Reproducibility safeguards

- Engines are split before any scaler or operating-condition model is fitted.
- Validation uses one deterministic observation cutoff per held-out engine.
- Validation and test have the same endpoint-level unit of analysis.
- Candidate training targets and reporting targets have separate caps.
- Candidate-ceiling and common-range scores are stored separately.
- Non-finite loss or prediction values fail the run; they are never imputed.
- Candidate matrices are complete before decisions can be locked.
- Lock and result files use atomic or create-only writes.
- Split variability, model-seed variability, and paired engine-bootstrap
  uncertainty are reported separately.
- The paired signed-rank calculation enumerates the exact sign-flip
  distribution, treats absolute differences at or below `1e-12` as numerical
  zeros, and uses average ranks for ties.
- The standard test-set plan is content-addressed, resumable, and not
  re-selectable.
- The release builder revalidates current result bytes and packages the exact
  preflight evidence set plus the result set named by the immutable plan;
  transient or unrelated files are excluded.
- The portable verifier remaps archived absolute source paths through
  `RELEASE_PROVENANCE.json` and checks the full plan/start/result/receipt
  chain after extraction.

## License

Code is released under the MIT License; see `LICENSE`. NASA C-MAPSS remains
subject to its source terms.
