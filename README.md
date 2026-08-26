# Empirical Cross-Grid Verification of Data-Driven Surrogate Models for Darcy Flow and Hyperelasticity

**Paper status:** Accepted in *Computers & Structures*.

## Overview

This study investigates how data-driven surrogate models behave when the grid used for training differs from the grid used for inference and evaluation.

Using the PDEBench DarcyFlow benchmark and a Mooney–Rivlin hyperelasticity benchmark, we evaluate Fourier Neural Operator (FNO), U-Net, and ResNet models over matched-grid and cross-grid training–evaluation paths.

The analysis goes beyond primary-field accuracy by combining field-, gradient-, energy-, spectral-, prediction-path-, and deformation-admissibility diagnostics. The main objective is finite-grid empirical verification: to determine whether conclusions drawn from matched-grid or primary-field accuracy remain valid across deployment grids and downstream quantities of interest.

## Key findings

* **DarcyFlow:** FNO showed relatively stable primary-field accuracy across the tested grids, but field-space proximity did not imply comparable proximity after applying the same Darcy flux-divergence transformation. For the FNO `128 → 256` path, the field-path distance was approximately `0.00434`, whereas the operator-response path distance was approximately `0.209`.

* **Hyperelasticity:** low displacement error did not guarantee low gradient error or deformation admissibility. For FNO trained on `128×32` and evaluated directly on the native `200×50` grid, the joint relative (L^2) error was approximately `0.0135`, while the relative vector (H^1)-seminorm error was approximately `1.42`. Every corresponding test prediction contained at least one cell with a non-positive predicted Jacobian determinant.

* **Overall:** cross-grid conclusions and architecture rankings depend on the diagnostic quantity, transfer path, and evaluation protocol. Primary-field accuracy alone is therefore insufficient for assessing cross-grid deployment.

## Repository contents

This repository provides the audited implementations, fixed data-split files, release configurations, and hash-verified processed figure and table data supporting the accepted manuscript.

To keep the release focused on minimum-sufficient reproducibility, the repository intentionally excludes:

* third-party raw datasets;
* model checkpoints;
* raw prediction tensors;
* full training logs;
* manuscript-automation utilities;
* obsolete implementations;
* historical experiments not used in the accepted study.

## Repository layout

* `src/`: models, data handling, metrics, training, and verification utilities.
* `configs/`: audited release configurations reconstructed from the final protocol records. These are not claimed to be byte-identical copies of every original run configuration file.
* `splits/`: byte-identical locked train/validation/test split assets.
* `results/`: hash-verified final processed figure and table data.
* `scripts/`: portable training, integrity-verification, manufactured-solution-audit, and convenience processed-data visualization entry points.
* `tests/`: release-integrity, model-forward, split, processed-result, spectral-protocol, and implementation-safety tests.
* `docs/`: dataset, figure-input, release-scope, and provenance guidance.

## Environment

### Confirmed final runtime

Final-run metadata confirms:

* Python `3.13.9`
* PyTorch `2.6.0+cu124`
* CUDA `12.4`

The original metadata did not preserve the exact version of every auxiliary Python package.

For a CUDA 12.4 environment, install the confirmed PyTorch build from the official PyTorch CUDA 12.4 package index, followed by the remaining dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124
python -m pip install "numpy>=1.26" "h5py>=3.10" "matplotlib>=3.8" "sympy>=1.13"
```

Alternatively, after installing the appropriate PyTorch build for your system:

```bash
python -m pip install -r requirements.txt
```

The non-PyTorch package versions above are minimum-compatible inferred requirements, not a claim that those exact versions were used for every final run.

On systems without a compatible NVIDIA CUDA 12.4 stack, install a CPU-compatible PyTorch build from the official PyTorch selector and then install the non-PyTorch dependencies. A CPU environment is suitable for integrity verification, tests, and lightweight inspection, but it is not hardware-identical to the final training runtime.

## Data placement

Third-party raw datasets are not redistributed in this repository.

Place them locally using the following structure:

```text
data/
├── darcy/
│   └── 2D_DarcyFlow_beta1.0_Train.hdf5
└── hyperelasticity/
    └── Hyperelasticity_n550_mooneyrivlin_200X50.npz
```

### DarcyFlow

Use the PDEBench DarcyFlow dataset with `beta = 1.0`.

The external Darcy HDF5 checksum was not retained in the final metadata. Obtain the dataset from PDEBench and verify its documented structure before training.

The supplied locked split file must be used exactly as provided:

```text
splits/darcy_beta1.0__seed0001__N8000_1000_1000.json
```

This corresponds to:

* 8000 training identities;
* 1000 validation identities;
* 1000 test identities.

Do not regenerate the split when reproducing the reported protocol.

### Hyperelasticity

The expected native dataset file is:

```text
Hyperelasticity_n550_mooneyrivlin_200X50.npz
```

Required SHA-256:

```text
E7D7821BD7295243F46C708BE360004408FA343D87536E25126BAEBE5704F599
```

Use the supplied locked split:

```text
splits/vino_hyperelasticity_split_seed0001_N400_75_75.json
```

This corresponds to:

* 400 training identities;
* 75 validation identities;
* 75 test identities.

The lower-resolution hyperelasticity representations are resampled from the same stored native `200×50` solutions; they are not independent finite-element solves on separately refined meshes.

Further dataset provenance and citation information is provided in:

```text
docs/DATASETS.md
```

## Verification and examples

### Verify all locked public processed data

```bash
python scripts/verify_processed_results.py
```

This checks the public processed-data files against the release manifest.

### Run the release test suite

```bash
python -m unittest discover -s tests -v
```

The tests cover, among other checks:

* public-module imports;
* locked split integrity;
* processed-result integrity;
* corrected FNO implementation gates;
* model forward-output shapes;
* manufactured-solution locked values;
* final spectral-protocol behavior.

### Run the Darcy manufactured-solution audit

```bash
python scripts/run_mms_audit.py
```

The audit evaluates the two prescribed smooth manufactured cases over the refinement sequence used in the study and checks the independent stencil implementation.

### Inspect the training interfaces

```bash
python scripts/train_darcy.py --help
python scripts/train_hyperelasticity.py --help
```

The release configurations are stored under:

```text
configs/darcy/
configs/hyperelasticity/
```

### Visualize processed data

For convenience, locked processed data can be rendered using:

```bash
python scripts/plot_processed_data_overview.py fig4 --output outputs/fig4.png
```

`plot_processed_data_overview.py` provides convenience visualizations of the released processed data.

It is **not** intended to reproduce the published figures, their exact panel composition, Origin formatting, or journal layout. The locked processed data are the authoritative public figure-support records.

Figure-specific data notes are available in:

```text
docs/FIGURE_SOURCES.md
```

The public:

```text
results/LOCKED_RESULTS_MANIFEST.csv
```

contains public-path integrity information only. Full source-location provenance is retained separately in the private release audit and is not exposed in this repository.

## Scope and interpretation

The following interpretation boundaries are important when using the released results:

* The study reports **finite-grid empirical verification**, not a test or proof of theoretical discretization invariance.

* `D_op` is a path-to-path relative distance obtained after applying the same Darcy `div(-a grad u)` transformation to two prediction paths. It is **not an absolute PDE residual** and should not be interpreted as direct proof of PDE fidelity or violation.

* Native/direct versus interpolated hyperelasticity comparisons are **sampled-grid comparisons**, not finite-element mesh-convergence studies.

* Three-training-run population standard deviations and paired bootstrap intervals summarize observed training-run and finite-test-sample variability. They are **not calibrated predictive uncertainty** or formal predictive-UQ guarantees.

* Architecture comparisons apply to the declared model definitions, training budgets, data partitions, grid paths, and checkpoint-selection protocols used in the study. They should not be interpreted as universal architecture rankings.

## FNO implementation audit

The public release contains only the final audited FNO implementations used for the revised study.

Each spectral layer retains both the positive and negative first-axis Fourier coefficient blocks and applies independent complex-valued linear maps to the two retained blocks.

Obsolete one-sided FNO implementations and routing are excluded from this repository.

Static implementation checks and model forward-shape tests are provided in:

```text
tests/
```

The released Darcy and hyperelasticity FNO implementations therefore correspond to the audited implementations used for the final reported results.

## Processed results

The `results/` directory contains the locked processed numerical data supporting the figures, tables, supplementary tables, manufactured-solution audit, validation-resolution sensitivity analysis, and selected bootstrap analyses.

Key groups include:

```text
results/
├── fig2/
├── fig3/
├── fig4/
├── fig5/
├── fig6/
├── fig7/
├── mms/
├── bootstrap/
├── validation_sensitivity/
└── tables/
```

These processed data are provided so that the reported numerical summaries can be inspected and independently visualized without distributing all original checkpoints, prediction tensors, or intermediate experiment artifacts.

## Citation

The manuscript has been accepted in *Computers & Structures*.

Publication DOI and final bibliographic metadata will be added after the article is formally published online.

For now, use the author and title metadata in:

```text
CITATION.cff
```

The repository intentionally does not invent or pre-assign a publication DOI.

## License

The code in this repository is released under the terms of the `LICENSE` file.

Third-party datasets remain subject to their original licenses, terms of use, and citation requirements.
