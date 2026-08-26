from __future__ import annotations
import csv, hashlib, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
    return h.hexdigest().upper()
class ProcessedResultTests(unittest.TestCase):
    def test_hashes(self):
        with (ROOT / "results/LOCKED_RESULTS_MANIFEST.csv").open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(rows),30)
        for row in rows:self.assertEqual(digest(ROOT/row["public_path"]),row["public_sha256"],row["public_path"])
    def test_key_row_counts(self):
        checks={"results/tables/Table1_benchmark_architecture_protocol.csv":6,"results/tables/Table2_darcy_high_resolution_physical_and_consistency.csv":19,"results/tables/Table3_hyperelasticity_native_and_admissibility.csv":6,"results/tables/Supplementary_Data_S1_full_spectral_profiles.csv":900,"results/tables/Supplementary_Data_S2_complete_bootstrap_outputs.csv":342,"results/mms/fd_manufactured_convergence_locked_copy.csv":12}
        for relative,expected in checks.items():
            with (ROOT / relative).open(encoding="utf-8-sig") as handle:
                self.assertEqual(sum(1 for _ in csv.DictReader(handle)), expected, relative)

    def test_fig3_public_data_policy(self):
        canonical = ROOT / "results/fig3/canonical_spatial_error_maps_eval256.npz"
        provenance = ROOT / "results/fig3/canonical_spatial_error_maps_eval256_provenance.json"
        self.assertTrue(canonical.is_file())
        self.assertTrue(provenance.is_file())
        self.assertFalse((ROOT / "results/fig3/avg_spatial_error_maps_eval256.npz").exists())
        self.assertFalse((ROOT / "results/fig3/corrected_avg_spatial_error_maps_eval256.npz").exists())
if __name__=="__main__":unittest.main()
