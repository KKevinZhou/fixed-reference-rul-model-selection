# External NASA benchmark

NASA raw files are deliberately excluded. Obtain the C-MAPSS Turbofan Engine Degradation Simulation data from the [official NASA PCoE page](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and its [listed download](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip). The listing was checked against the official page on 2026-10-18. The retained original archive and individual file checksums are in archive/source/canonical_cmapss_sha256.json.

For retraining or fresh weight inference, extract the nested CMAPSSData archive and place its twelve train/test/RUL files under:

```text
YOUR_DATA_ROOT/Experiment_FD001/CMAPSSData/train_FD001.txt
YOUR_DATA_ROOT/Experiment_FD001/CMAPSSData/test_FD001.txt
YOUR_DATA_ROOT/Experiment_FD001/CMAPSSData/RUL_FD001.txt
... corresponding FD002, FD003, FD004 files ...
```

Pass YOUR_DATA_ROOT explicitly through --data-root. The tool checks the individual file SHA-256 values before reading them. No raw data are needed for prediction-based table recomputation; ordered target-payload hashes verify the labels embedded in the derived predictions.

NASA dataset citation: A. Saxena and K. Goebel (2008), Turbofan Engine Degradation Simulation Data Set, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
