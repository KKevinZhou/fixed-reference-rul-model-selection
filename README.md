# Fixed-Reference Scoring for RUL Model Selection

Code and derived predictions for **Fixed-Reference Scoring for Task-Consistent RUL Model Selection under Tunable Target Capping**.

Authors: Dejian Jiang, Zhiheng Zhou, and Xiyue Deng.

## Reproduce the reported analyses

The reviewed `2026.10.18-rc1` artifact is preserved in [`reproduction/`](reproduction/). It contains source code, configurations, 210 successful result records (320 fits), derived predictions, 13 manuscript summary tables, and eight predefined checkpoint-audit cases. Raw NASA data and preprocessed sensor features are excluded. See [NASA_DATA.md](reproduction/NASA_DATA.md) for official links and checksums.

Use Python 3.13 and install the dependencies in a dedicated environment. Run from the reproduction directory:

```shell
cd reproduction
python -m pip install -r requirements.txt
python public.py verify
python public.py recompute --output ../recomputed
```

Recomputing the reported summaries requires no raw NASA data. Fresh inference and training require externally obtained, hash-checked NASA data; see the [artifact instructions](reproduction/README.md). The eight distributed weights reproduced 1,414 archived inference outputs exactly in the recorded environment. Cross-hardware equality is not guaranteed.

## Scope and interpretation

The standard test data were inspected during earlier development. These results are conditional descriptive evidence. Fixed-reference scoring is not consistently better than fixed cap125; model-seed tests do not establish conventional significance; the 30-cycle delay proxy does not improve. The original-precision FD002 cap125 failure and separate FP32-attention sensitivity remain distinct.

The files under `reproduction/` are the byte-preserved upload candidate. Its publication-status files describe the state when the snapshot was built; they are historical build metadata, not the live repository status. The Zenodo archive is being prepared and has not yet been published. No DOI is claimed here.

## Licenses and attribution

Author software and software documentation retain the existing [MIT license](LICENSE). Author-derived predictions, configuration datasets, and the eight selected weights use [CC BY 4.0](reproduction/DATA_AND_WEIGHTS_LICENSE.md). NASA and other third-party rights remain separate. See [third-party notices](reproduction/THIRD_PARTY_NOTICES.md) and [AI assistance record](reproduction/AI_ASSISTANCE_RECORD.md).

The protected manuscript BibTeX file and scientific result bytes were not edited for this upload.
