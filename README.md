# Empirical Cross-Grid Verification of Data-Driven Surrogate Models for Darcy Flow and Hyperelasticity

**Paper status:** Accepted in *Computers & Structures*.

## Overview

This repository accompanies the accepted manuscript **“Empirical Cross-Grid Verification of Data-Driven Surrogate Models for Darcy Flow and Hyperelasticity.”**

The study investigates how data-driven surrogate models behave when the grid used for training differs from the grid used for inference and evaluation.

Using the PDEBench DarcyFlow benchmark and a Mooney–Rivlin hyperelasticity benchmark, we evaluate Fourier Neural Operator (FNO), U-Net, and ResNet models over matched-grid and cross-grid training–evaluation paths.

The analysis goes beyond primary-field accuracy by combining:

* field-level errors;
* gradient-sensitive errors;
* energy-related diagnostics;
* spatial error maps;
* spectral residual analysis;
* prediction-path diagnostics;
* deformation-admissibility checks.

The objective is **finite-grid empirical verification**: to determine whether conclusions drawn from matched-grid or primary-field accuracy remain valid across deployment grids and downstream quantities of interest.

## Key findings

* **DarcyFlow:** FNO showed relatively stable primary-field accuracy across the tested grids, but field-space proximity did not imply comparable proximity after applying the same Darcy flux-divergence transformation. For the FNO `128 → 256` path, the field-path distance was approximately `0.00434`, whereas the operator-response path distance was approximately `0.209`.

* **Hyperelasticity:** low displacement error did not guarantee low gradient error or deformation admissibility. For FNO trained on `128×32` and evaluated directly on the native `200×50` grid, the joint relative L2 error was approximately `0.0135`, while the relative vector H1-seminorm error was approximately `1.42`. Every corresponding test prediction contained at least one cell with a non-positive predicted Jacobian determinant.

* **Overall:** cross-grid conclusions and architecture rankings depended on the diagnostic quantity, transfer path, and evaluation protocol. Primary-field accuracy alone was therefore insufficient for characterizing cross-grid deployment behavior.

## Repository contents

This repository provides the public research assets supporting the accepted manuscript:

* final audited model implementations;
* fixed train/validation/test split files;
* release configurations;
* metric and verification implementations;
* hash-verified processed figure and table data;
* numerical-verification utilities;
* release-integrity tests.

The release follows a **minimum-sufficient reproducibility** principle.

It intentionally does **not** redistribute:

* third-party raw datasets;
* model checkpoints;
* full prediction tensors;
* complete training logs;
* manuscript-automation utilities;
* obsolete implementations;
* historical experiments not used in the accepted study.

## Quick start

After installing the environment and placing the external datasets as described below, the release can be checked with:

```bash
python -m unittest discover -s tests -v
python scripts/verify_processed_results.py
python scripts/run_mms_audit.py
```

To inspect the available training interfaces:

```bash
python scripts/train_darcy.py --help
python scripts/train_hyperelasticity.py --help
```

The released processed numerical results can be inspected and visualized **without retraining the models**.

## Installation

### Confirmed final training runtime

Final-run metadata confirms the following core runtime:

* Python `3.13.9`
* PyTorch `2.6.0+cu124`
* CUDA `12.4`

The original metadata did not preserve the exact version of every auxiliary Python package.

For a CUDA 12.4 environment:

```bash
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124
python -m pip install -r requirements.txt
```

The non-PyTorch dependencies in `requirements.txt` are minimum-compatible inferred requirements rather than a claim that those exact versions were used for every final run.

On systems without a compatible NVIDIA CUDA 12.4 stack, install an appropriate CPU-compatible PyTorch build from the official PyTorch installation selector and then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

A CPU environment is suitable for integrity verification, unit tests, numerical audits, and lightweight inspection, but it is not hardware-identical to the final training environment.

## Data

Third-party raw datasets are not redistributed in this repository.

Place the required datasets locally using the following structure:

```text
data/
├── darcy/
│   └── 2D_DarcyFlow_beta1.0_Train.hdf5
└── hyperelasticity/
    └── Hyperelasticity_n550_mooneyrivlin_200X50.npz
```

### DarcyFlow

Use the **PDEBench DarcyFlow dataset with `beta = 1.0`**.

Expected local file:

```text
data/darcy/2D_DarcyFlow_beta1.0_Train.hdf5
```

The external Darcy HDF5 checksum was not retained in the final metadata. Obtain the dataset from PDEBench and verify its documented structure before training.

Use the supplied fixed split:

```text
splits/darcy_beta1.0__seed0001__N8000_1000_1000.json
```

The split contains:

* 8000 training identities;
* 1000 validation identities;
* 1000 test identities.

To reproduce the reported protocol, use this split exactly as supplied rather than regenerating a new partition.

### Hyperelasticity

Expected native dataset file:

```text
data/hyperelasticity/Hyperelasticity_n550_mooneyrivlin_200X50.npz
```

For exact file verification, the expected SHA-256 checksum is:

```text
E7D7821BD7295243F46C708BE360004408FA343D87536E25126BAEBE5704F599
```

Use the supplied fixed split:

```text
splits/vino_hyperelasticity_split_seed0001_N400_75_75.json
```

The split contains:

* 400 training identities;
* 75 validation identities;
* 75 test identities.

The lower-resolution hyperelasticity representations are resampled from the same stored native `200×50` solutions. They are therefore used for **sampled-grid transfer assessment**, not as independent finite-element solutions obtained from separate mesh-refinement studies.

Further dataset provenance and citation information is available in:

```text
docs/DATASETS.md
```

## Processed results

The `results/` directory contains the processed numerical data supporting the main figures, tables, supplementary analyses, and numerical-verification results reported in the study.

These data can be inspected and visualized without retraining the models.

| Location                          | Contents                                                                    |
| --------------------------------- | --------------------------------------------------------------------------- |
| `results/fig2/`                   | DarcyFlow training–evaluation relative L2 matrices                          |
| `results/fig3/`                   | Test-set-averaged Darcy spatial-error maps                                  |
| `results/fig4/`                   | Darcy spectral residual profiles and frequency-band summaries               |
| `results/fig5/`                   | Darcy field-path and operator-response path diagnostics                     |
| `results/fig6/`                   | Hyperelasticity displacement and gradient metrics                           |
| `results/fig7/`                   | Native/interpolated, directional, and deformation-admissibility diagnostics |
| `results/mms/`                    | Darcy manufactured-solution convergence audit                               |
| `results/validation_sensitivity/` | Darcy validation-resolution sensitivity results                             |
| `results/bootstrap/`              | Selected paired-bootstrap outputs                                           |
| `results/tables/`                 | Consolidated main and supplementary table data                              |

The processed files are the authoritative public numerical records supporting the released figure and table values.

Large intermediate research artifacts such as checkpoints, raw prediction tensors, and complete experiment directories are intentionally not distributed.

To verify the public processed files against the release manifest:

```bash
python scripts/verify_processed_results.py
```

Figure-specific data notes are provided in:

```text
docs/FIGURE_SOURCES.md
```

The public manifest:

```text
results/LOCKED_RESULTS_MANIFEST.csv
```

contains public-path integrity information only. Full internal source-location provenance is retained separately from the public repository.

## Repository layout

```text
.
├── configs/
├── docs/
├── results/
├── scripts/
├── splits/
├── src/
├── tests/
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt
```

### `src/`

Core research implementation, including:

* models;
* data handling;
* metrics;
* training utilities;
* numerical-verification utilities.

### `configs/`

Audited release configurations reconstructed from the final protocol records.

These files are intended to represent the final released protocol but are not claimed to be byte-identical copies of every original run configuration file.

### `splits/`

Byte-identical locked train/validation/test split assets used by the released protocol.

### `results/`

Hash-verified processed numerical results supporting the reported figures, tables, supplementary analyses, and numerical audits.

### `scripts/`

Portable entry points for:

* training-interface access;
* processed-result integrity verification;
* Darcy manufactured-solution audit;
* convenience visualization of processed data.

### `tests/`

Release tests covering:

* public-module imports;
* split integrity;
* processed-result integrity;
* FNO implementation safety;
* model forward shapes;
* manufactured-solution locked values;
* spectral-protocol behavior.

### `docs/`

Additional documentation covering:

* dataset provenance;
* figure-support data;
* release scope and interpretation.

## Verification and reproducibility

### Run the complete release test suite

```bash
python -m unittest discover -s tests -v
```

The tests include checks for:

* public-module importability;
* fixed split integrity;
* processed-result integrity;
* absence of obsolete FNO routing;
* corrected FNO forward behavior;
* expected model-output shapes;
* manufactured-solution locked values;
* final Darcy spectral-protocol behavior.

### Verify processed-result integrity

```bash
python scripts/verify_processed_results.py
```

This checks the public processed-data files against their release hashes.

### Run the Darcy manufactured-solution audit

```bash
python scripts/run_mms_audit.py
```

The audit reproduces the two prescribed smooth manufactured cases over the refinement sequence used in the study and checks the independent stencil implementation.

### Inspect training interfaces

```bash
python scripts/train_darcy.py --help
python scripts/train_hyperelasticity.py --help
```

Release configurations are located under:

```text
configs/darcy/
configs/hyperelasticity/
```

### Convenience visualization of processed data

For example:

```bash
python scripts/plot_processed_data_overview.py fig4 --output outputs/fig4.png
```

`plot_processed_data_overview.py` provides convenience visualizations of the released processed numerical data.

It is **not** intended to reproduce the published figures, their exact panel composition, Origin formatting, or journal layout.

The numerical files in `results/` are the authoritative public figure-support data.

## Scope and interpretation

The following interpretation boundaries are important when using the released code and results.

* The study reports **finite-grid empirical verification**. It does not test or prove theoretical discretization invariance.

* `D_op` is a path-to-path relative distance obtained after applying the same Darcy `div(-a grad u)` transformation to two prediction paths. It is **not an absolute PDE residual** and should not be interpreted as direct proof of PDE fidelity or violation.

* The Darcy manufactured-solution audit verifies the numerical implementation of the released stencil for the prescribed smooth manufactured cases. It does not establish convergence of the complete PDEBench source/boundary-value problem.

* Native/direct versus interpolated hyperelasticity comparisons are **sampled-grid comparisons**, not finite-element mesh-convergence studies.

* Three-training-run population standard deviations summarize observed training-run variability.

* Paired bootstrap intervals summarize finite-test-sample variability after the specified aggregation procedure.

* Neither the three-run statistics nor the bootstrap intervals constitute calibrated predictive uncertainty or a formal predictive-UQ guarantee.

* Architecture comparisons apply to the declared model definitions, data partitions, grid paths, training budgets, and checkpoint-selection protocols used in the study. They should not be interpreted as universal architecture rankings.

## FNO implementation audit

The public release contains only the final audited FNO implementations used for the revised study.

Each spectral layer retains both the positive and negative first-axis Fourier coefficient blocks and applies independent complex-valued linear maps to the two retained blocks.

Obsolete one-sided FNO implementations and routing are excluded from this repository.

Static implementation checks and model forward-shape tests are provided under:

```text
tests/
```

The released Darcy and hyperelasticity FNO implementations therefore correspond to the audited implementations used for the final reported results.

## Citation

The manuscript has been **accepted in *Computers & Structures***.

The final DOI and publication metadata will be added after the article is formally published online.

Until then, author and title metadata are available in:

```text
CITATION.cff
```

This repository intentionally does not invent or pre-assign a publication DOI.

## License

The code in this repository is released under the terms specified in:

```text
LICENSE
```

Third-party datasets remain subject to their original licenses, terms of use, and citation requirements.
