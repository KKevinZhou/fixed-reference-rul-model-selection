# Fixed-Reference Scoring for RUL Model Selection

Code and study configurations for **Fixed-Reference Scoring for Task-Consistent RUL Model Selection under Tunable Target Capping**.

Authors: Dejian Jiang, Zhiheng Zhou, and Xiyue Deng.

## Code and full reproduction artifact

This GitHub repository hosts the software and configuration subset of the reviewed `2026.10.18-rc1` artifact. It includes the training, inference, verification, and summary-generation code under [reproduction/](reproduction/). Derived prediction records and checkpoint weight binaries are archived separately on [Zenodo](https://zenodo.org/records/23464274) and are not included in this code checkout. The version archive was published on 2026-10-19: **[10.5281/zenodo.23464274](https://doi.org/10.5281/zenodo.23464274)**.

The complete reviewed archive contains 210 successful result records (320 fits), derived predictions, thirteen manuscript summary tables, and eight predefined checkpoint-audit cases. It excludes raw NASA files and preprocessed sensor features. [NASA_DATA.md](reproduction/NASA_DATA.md) supplies official links and checksums. The included [PUBLIC_MANIFEST.json](reproduction/PUBLIC_MANIFEST.json) describes the complete artifact, not just this code checkout.

## Reproduction

Download the [reviewed ZIP](https://zenodo.org/records/23464274/files/Fixed_Reference_Public_20261018_RC1.zip?download=1) and extract its contents directly into `reproduction/` to restore all manifest-listed files. The ZIP has no enclosing directory. Use Python 3.13 and a dedicated dependency environment. Then run:

```shell
cd reproduction
python -m pip install -r requirements.txt
python public.py verify
python public.py recompute --output ../recomputed
```

Recomputing the reported summaries requires no raw NASA data. Fresh inference and training require separately obtained, hash-checked NASA data. See the code's command-line help and the full archive instructions. The eight distributed weights reproduced 1,414 archived inference outputs exactly in the recorded environment; cross-hardware equality is not guaranteed.

## Interpretation

The standard test data were inspected during earlier development. These results are conditional descriptive evidence. Fixed-reference scoring is not consistently better than fixed cap125; model-seed tests do not establish conventional significance; the 30-cycle delay proxy does not improve. The original-precision FD002 cap125 failure and separate FP32-attention sensitivity remain distinct. Retained preparation records describe historical build state.

## Licenses

Author software and software documentation retain the existing [MIT license](LICENSE). Author-derived predictions, configuration datasets, and the eight selected weights use [CC BY 4.0](reproduction/DATA_AND_WEIGHTS_LICENSE.md). NASA and other third-party rights remain separate. See [third-party notices](reproduction/THIRD_PARTY_NOTICES.md) and the [AI assistance record](reproduction/AI_ASSISTANCE_RECORD.md).

The protected manuscript BibTeX file and scientific result bytes were not edited for publication.

Archived ZIP SHA-256: `fe07c9929fd4b908eca04407ca6a670b69f2bc54464e4db7448ab5ffc5db8fa6` (local reviewed artifact); the public record reports the matching MD5 `958ce62ab07f6162a84169cb7a5c5699`.

Reviewed code snapshot: [`384694c`](https://github.com/KKevinZhou/fixed-reference-rul-model-selection/commit/384694c0499f604fa76e40c619cfbcc909fa91e4). Later metadata-only commits supply the archival links.
