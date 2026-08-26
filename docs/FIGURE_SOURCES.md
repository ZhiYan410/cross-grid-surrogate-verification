# Locked Figure-Support Data

The files under `results/` are the authoritative public processed data that
support the published figures and tables. They are hash-verified by
`scripts/verify_processed_results.py`.

`scripts/plot_processed_data_overview.py` is a convenience visualization tool.
It is not intended to reproduce the published figures, their exact panel
composition, or journal layout.

| Paper item | Authoritative public processed data |
|---|---|
| Fig. 2 | `results/fig2/TableS1_fno_darcy_full_physical_matrix.csv`, `results/fig2/TableS2_unet_resnet_darcy_full_physical_matrices.csv` |
| Fig. 3 | `results/fig3/canonical_spatial_error_maps_eval256.npz` and `results/fig3/canonical_spatial_error_maps_eval256_provenance.json` |
| Fig. 4 | `results/fig4/Supplementary_Data_S1_full_spectral_profiles.csv`, `results/fig4/TableS3_darcy_spectral_band_summaries.csv` |
| Fig. 5 | `results/fig5/TableS4_darcy_path_consistency_amplification_and_fd_baseline.csv` |
| Fig. 6 | `results/fig6/TableS6_hyperelasticity_complete_grid_summary.csv` |
| Fig. 7 | `results/fig7/TableS7_hyperelasticity_directional_and_direct_interpolated.csv`, `results/fig7/Table3_hyperelasticity_native_and_admissibility.csv` |
| MMS | `results/mms/fd_manufactured_convergence_locked_copy.csv` |
| Validation sensitivity | `results/validation_sensitivity/TableS5_darcy_validation_protocol_sensitivity.csv` |
| Bootstrap | `results/bootstrap/Supplementary_Data_S2_complete_bootstrap_outputs.csv`, `results/bootstrap/TableS8_key_bootstrap_comparisons_ci95.csv` |

For Fig. 3, the canonical NPZ is the sole public plotting dataset. The two
upstream map artifacts used to produce it are archived privately; their
filenames, hashes, and deterministic composition are recorded in the public
provenance JSON.
