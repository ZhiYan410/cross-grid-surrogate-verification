# Empirical Cross-Grid Verification of Data-Driven Surrogate Models for Darcy Flow and Hyperelasticity

This release candidate is a minimum-sufficient reproducibility package for the
accepted-paper results. It contains final audited implementations, immutable
split files, release configurations, and hash-verified processed figure and
table data. It deliberately excludes third-party datasets, checkpoints, raw
prediction tensors, training logs, manuscript automation, and historical
experiments.

## Scope

- The study reports finite-grid empirical verification, not theoretical
  discretization invariance.
- `D_op` is a path-to-path relative distance after applying the same Darcy
  `div(-a grad u)` transformation. It is not an absolute PDE residual.
- Native/direct versus interpolated hyperelasticity comparisons are sampled-grid
  comparisons, not finite-element mesh convergence.
- Three-training-run population SD and paired bootstrap intervals summarize
  observed run/sample variability; they are not calibrated predictive
  uncertainty.

## Corrected FNO gate

Only corrected FNO implementations are included. Each FNO retains both the
positive and negative first-axis Fourier blocks from `rfft2` and applies
independent complex-valued linear maps to those blocks. No obsolete one-sided
FNO route is exposed. The static and forward-shape gates are in `tests/`.

## Repository layout

- `src/`: models, data handling, metrics, training, and verification utilities.
- `configs/`: audited release configurations reconstructed from final protocol
  records; they are not byte-identical original run configuration files.
- `splits/`: byte-identical locked split assets.
- `results/`: hash-verified final processed figure/table data.
- `scripts/`: portable training, integrity verification, MMS audit, and
  convenience processed-data overview entrypoints.
- `docs/`: dataset, figure-input, and provenance guidance.

## Environment

### Confirmed final runtime

Final run metadata confirms Python `3.13.9`, PyTorch `2.6.0+cu124`, and CUDA
`12.4`. The original metadata did not preserve every package version.

For a CUDA 12.4 environment, install the confirmed PyTorch wheel from the
official PyTorch CUDA 12.4 index, then install the remaining dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124
python -m pip install "numpy>=1.26" "h5py>=3.10" "matplotlib>=3.8" "sympy>=1.13"
```

The non-PyTorch packages above are minimum-compatible inferred requirements,
not a claim that those exact versions were used for every final run. On systems
without a compatible NVIDIA CUDA 12.4 stack, install a CPU-compatible PyTorch
build from the official PyTorch selector and then install the non-PyTorch
dependencies. A CPU environment is suitable for verification and smoke tests,
but is not hardware-identical to the final training runtime.

## Data placement

Do not commit third-party datasets. Place them locally and pass paths
explicitly:

```text
data/
  darcy/2D_DarcyFlow_beta1.0_Train.hdf5
  hyperelasticity/Hyperelasticity_n550_mooneyrivlin_200X50.npz
```

The hyperelasticity filename has required SHA-256
`E7D7821BD7295243F46C708BE360004408FA343D87536E25126BAEBE5704F599`.
The external Darcy HDF5 checksum was not retained in final metadata; obtain the
beta=1.0 DarcyFlow dataset from PDEBench and verify its documented structure
before training. Apply the supplied split files exactly; do not regenerate
splits.

Dataset provenance and citations are in [docs/DATASETS.md](docs/DATASETS.md).

## Verification and examples

```powershell
# Verify every locked public processed-data file against its release hash.
python scripts/verify_processed_results.py

# Static safety, locked-data, MMS, and model-forward smoke tests.
python -m unittest discover -s tests -v

# Run the full two-case manufactured-stencil convergence audit.
python scripts/run_mms_audit.py

# Train a new corrected Darcy FNO run in a new output directory.
python scripts/train_darcy.py --h5 data/darcy/2D_DarcyFlow_beta1.0_Train.hdf5 --split splits/darcy_beta1.0__seed0001__N8000_1000_1000.json --architecture corrected_fno --train-resolution 128 --seed 1 --output runs/darcy_fno_128_seed1

# Render a convenience overview of locked Fig. 4 processed data.
python scripts/plot_processed_data_overview.py fig4 --output outputs/fig4.png
```

`plot_processed_data_overview.py` provides convenience visualizations of
locked processed data. It is not intended to reproduce the published figures,
their exact panel composition, or journal layout. The locked processed data are
the authoritative public figure-support records. Figure-specific data notes are
in [docs/FIGURE_SOURCES.md](docs/FIGURE_SOURCES.md).

`results/LOCKED_RESULTS_MANIFEST.csv` is intentionally public-path-only. The
private release audit retains the full source-location provenance map without
exposing internal work directories in this package.

## Citation

Use the author and title metadata in `CITATION.cff`. Preferred-citation metadata
will be finalized when a publication DOI is assigned; this release does not
invent a DOI.
