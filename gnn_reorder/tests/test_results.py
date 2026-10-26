"""CPU-only regression checks for saved results and CSV persistence.

Run from the repository root with:
    python3 -m unittest discover -s gnn_reorder/tests -v
"""

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, PROJECT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timer = load_module("project_timer", "profiling/timer.py")
compare = load_module("project_compare", "compare_results.py")


class TimerCSVTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "timing.csv"
        self.timer = timer.EpochTimer()
        self.timer._times = [1.0, 2.0]

    def test_phase_specific_columns_stay_aligned(self):
        self.timer.save_csv(str(self.path), {
            "dataset": "arxiv", "mode": "full-batch", "k": "N/A",
            "train_acc": 0.5, "val_acc": 0.6,
        })
        self.timer.save_csv(str(self.path), {
            "dataset": "arxiv", "k": 4, "train_acc": 0.7, "val_acc": 0.8,
        })
        self.timer.save_csv(str(self.path), {"dataset": "products", "extra": 9})
        with self.path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["mode"], "full-batch")
        self.assertEqual(rows[1]["mode"], "")
        self.assertEqual(rows[1]["k"], "4")
        self.assertEqual(rows[1]["train_acc"], "0.7")
        self.assertEqual(rows[1]["val_acc"], "0.8")
        self.assertEqual(rows[0]["extra"], "")
        self.assertEqual(rows[2]["extra"], "9")

    def test_malformed_existing_rows_are_rejected_without_changes(self):
        for content in ("dataset,model\narxiv\n", "dataset,model\narxiv,SAGE,extra\n"):
            with self.subTest(content=content):
                self.path.write_text(content)
                with self.assertRaisesRegex(ValueError, "Malformed CSV"):
                    self.timer.save_csv(str(self.path), {"dataset": "arxiv"})
                self.assertEqual(self.path.read_text(), content)


class SavedResultTests(unittest.TestCase):
    def test_saved_csv_columns_and_accuracy_mapping(self):
        for name in ("timing.csv", "phase3_results.csv"):
            with self.subTest(name=name):
                with (PROJECT / "results" / name).open(newline="") as f:
                    raw = list(csv.reader(f))
                self.assertGreater(len(raw), 1)
                self.assertTrue(all(len(row) == len(raw[0]) for row in raw[1:]))
                for row in compare.load_csv(PROJECT / "results" / name):
                    self.assertIn(row["dataset"], ("ogbn-arxiv", "ogbn-products"))
                    self.assertIn(row["model"], ("GraphSAGE", "GAT"))
                    for key in ("train_acc", "val_acc"):
                        value = row.get(key, "")
                        if value and value.lower() != "nan":
                            self.assertTrue(0 <= float(value) <= 1)
        rows = compare.load_csv(PROJECT / "results" / "timing.csv")
        rcm = next(r for r in rows if r["model"] == "GraphSAGE" and r["reordering"] == "rcm")
        self.assertEqual(rcm["mode"], "full-batch")
        self.assertEqual(rcm["k"], "N/A")
        self.assertEqual(rcm["train_acc"], "0.5793")
        self.assertEqual(rcm["val_acc"], "0.5721")

    def test_phase3_preferred_without_dropping_other_configurations(self):
        baseline = {"dataset": "arxiv", "model": "SAGE", "reordering": "baseline", "mean_ms_per_epoch": "10"}
        rcm = {**baseline, "reordering": "rcm"}
        enriched = {**baseline, "mean_ms_per_epoch": "9", "temporal_reuse_ratio": "80"}
        hybrid = {**enriched, "reordering": "hybrid"}
        merged = compare.merge_rows([baseline, rcm], [enriched, hybrid])
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0], enriched)
        self.assertIn(rcm, merged)
        self.assertIn(hybrid, merged)


if __name__ == "__main__":
    unittest.main()
