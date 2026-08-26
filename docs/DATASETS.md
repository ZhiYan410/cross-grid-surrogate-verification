# External Datasets

No third-party raw data are redistributed in this repository.

## PDEBench DarcyFlow beta=1.0

- Source: PDEBench repository and DaRUS PDEBench Datasets collection.
- Required local filename: `2D_DarcyFlow_beta1.0_Train.hdf5`.
- Citation: Takamoto et al., *PDEBench: An Extensive Benchmark for Scientific Machine Learning*, NeurIPS Datasets and Benchmarks 2022, https://arxiv.org/abs/2210.07182.
- Dataset collection: https://doi.org/10.18419/darus-2986.
- The final training code expects HDF5 keys `nu`, `tensor`, `x-coordinate`, and `y-coordinate`, with 10,000 sample identities. Apply `splits/darcy_beta1.0__seed0001__N8000_1000_1000.json` unchanged.

## VINO Mooney-Rivlin hyperelastic beam

- Source: VINO project/data distribution associated with Eshaghi et al., *Variational Physics-informed Neural Operator for solving partial differential equations*, CMAME 2025, https://doi.org/10.1016/j.cma.2025.117785.
- Required local filename: `Hyperelasticity_n550_mooneyrivlin_200X50.npz`.
- Required SHA-256: `E7D7821BD7295243F46C708BE360004408FA343D87536E25126BAEBE5704F599`.
- The final loader expects NPZ keys `traction` and `disp2D`, with shapes `(550, 50)` and `(550, 50, 200, 2)`. Apply `splits/vino_hyperelasticity_split_seed0001_N400_75_75.json` unchanged.

The original VINO repository is not vendored here. This release contains only the paper-specific data-driven model/metric implementation and cites the external source.
