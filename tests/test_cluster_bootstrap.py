"""Artificial paired timing differences only; NO final optimizer experiments."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from opttheory.experiments.analysis import bootstrap_dataset_time_difference
from opttheory.experiments.evaluation import validate_evaluation_protocol


ROOT = Path(__file__).resolve().parents[1]


class DatasetClusterBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((ROOT / "config/evaluation_protocol.json").read_text())

    @staticmethod
    def _two_dataset_fixture():
        # Each dataset's five starts have ALREADY been grouped and averaged.
        return {
            "independent_dataset_count": 2,
            "dataset_mean_newton_minus_gd_seconds": 0.3,
            "dataset_records": [
                {"dataset_id": "P01", "paired_starts": 5,
                 "mean_newton_minus_gd_seconds": 0.2},
                {"dataset_id": "P02", "paired_starts": 5,
                 "mean_newton_minus_gd_seconds": 0.4},
            ],
        }

    def _test_protocol(self):
        protocol = deepcopy(self.protocol)
        protocol["analysis"]["dataset_seeds_per_model"] = 2  # unit fixture only
        return protocol

    def test_bootstrap_resamples_whole_datasets_and_not_individual_starts(self):
        interval = bootstrap_dataset_time_difference(
            "exponential_decay", self._two_dataset_fixture(), self._test_protocol())
        self.assertEqual(interval["independent_dataset_count"], 2)
        self.assertFalse(interval["starting_points_resampled_as_independent_units"])
        self.assertEqual(interval["resamples"], 10000)
        self.assertEqual(interval["confidence_level"], 0.95)
        self.assertAlmostEqual(interval["point_estimate_seconds"], 0.3)
        # Sampling two whole dataset means {0.2, 0.4} yields only
        # averages {0.2, 0.3, 0.4}, not the extreme individual-start times.
        self.assertAlmostEqual(interval["lower_seconds"], 0.2)
        self.assertAlmostEqual(interval["upper_seconds"], 0.4)
        self.assertIsNone(interval["p_value"])

    def test_fixed_model_seed_and_input_order_make_results_reproducible(self):
        summary = self._two_dataset_fixture()
        protocol = self._test_protocol()
        first = bootstrap_dataset_time_difference("logistic", summary, protocol)
        repeated = bootstrap_dataset_time_difference("logistic", summary, protocol)
        self.assertEqual(first, repeated)
        reversed_summary = deepcopy(summary)
        reversed_summary["dataset_records"].reverse()
        reordered = bootstrap_dataset_time_difference("logistic", reversed_summary, protocol)
        self.assertEqual(first, reordered)
        self.assertEqual(first["model_seed_offset"], 1)

    def test_mutated_or_incomplete_clusters_cannot_create_fake_confidence(self):
        summary = self._two_dataset_fixture()
        protocol = self._test_protocol()
        changed = deepcopy(protocol)
        changed["analysis"]["bootstrap"]["resampling_unit"] = "individual_start"
        with self.assertRaisesRegex(ValueError, "whole dataset"):
            bootstrap_dataset_time_difference("logistic", summary, changed)
        changed = deepcopy(protocol)
        changed["analysis"]["bootstrap"]["resamples"] = 9
        with self.assertRaisesRegex(ValueError, "10,000"):
            bootstrap_dataset_time_difference("logistic", summary, changed)
        incomplete = deepcopy(summary)
        incomplete["dataset_records"].pop()
        with self.assertRaisesRegex(ValueError, "all distinct"):
            bootstrap_dataset_time_difference("logistic", incomplete, protocol)

    def test_real_protocol_approves_fixed_95_percent_cluster_settings(self):
        plan = json.loads((ROOT / "config/experiment_plan.json").read_text())
        frozen = json.loads((ROOT / "config/selected_gd_rates.json").read_text())
        manifests = validate_evaluation_protocol(plan, frozen, self.protocol)
        self.assertEqual(manifests["exponential_decay"]["timed_invocations"], 750)
        self.assertEqual(manifests["logistic"]["timed_invocations"], 750)
        changed = deepcopy(self.protocol)
        changed["analysis"]["bootstrap"]["resampling_unit"] = "individual_start"
        with self.assertRaisesRegex(ValueError, "cluster-bootstrap"):
            validate_evaluation_protocol(plan, frozen, changed)


if __name__ == "__main__":
    unittest.main()
