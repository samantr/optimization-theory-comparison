"""Fast actual-GD calibration checks on a reduced test-only grid."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from opttheory.datasets.synthetic import generate_model1, generate_model2
from opttheory.experiments.run_calibration import run_model_calibration


ROOT = Path(__file__).resolve().parents[1]


class CalibrationExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((ROOT / "config/experiment_plan.json").read_text())

    def _quick_plan(self):
        config = deepcopy(self.plan)
        tuning = config["gradient_descent_learning_rate_tuning"]
        tuning["calibration_datasets_per_model"] = 1
        tuning["candidate_learning_rates"] = [0.01, 0.03]
        tuning["calibration_noise_seeds_by_model"] = {
            "exponential_decay": [2000], "logistic": [2100],
        }
        config["optimizer_stopping"]["max_iterations_by_method"]["gradient_descent"] = 1
        return config

    def test_one_seed_two_rates_runs_for_both_models(self):
        plan = self._quick_plan()
        for model, number in (("exponential_decay", 1), ("logistic", 2)):
            with self.subTest(model=model):
                prototype = json.loads((ROOT / f"config/model{number}.json").read_text())
                outcome = run_model_calibration(plan, model, prototype)
                self.assertEqual(outcome["registered_conditions"], 10)
                self.assertEqual(len(outcome["runs"]), 10)
                self.assertIn(outcome["chosen_learning_rate"], [0.01, 0.03])
                self.assertEqual(set(outcome["dataset_hashes_by_noise_seed"]),
                                 {"2000" if number == 1 else "2100"})
                self.assertTrue(all(row["split_seed"] == 2026 for row in outcome["runs"]))

    def test_changing_test_targets_cannot_change_selected_rate_or_losses(self):
        plan = self._quick_plan()
        for model, number, original in (
            ("exponential_decay", 1, generate_model1),
            ("logistic", 2, generate_model2),
        ):
            with self.subTest(model=model):
                prototype = json.loads((ROOT / f"config/model{number}.json").read_text())
                untouched = run_model_calibration(plan, model, prototype)

                def corrupt_only_test_labels(cfg):
                    data = original(cfg)
                    data.y_observed[data.split == "test"] += 1_000_000.0
                    return data

                with patch("opttheory.experiments.run_calibration.generate_model"
                           + str(number), side_effect=corrupt_only_test_labels):
                    altered = run_model_calibration(plan, model, prototype)

                self.assertEqual(untouched["chosen_learning_rate"],
                                 altered["chosen_learning_rate"])
                self.assertEqual(untouched["rate_scores"], altered["rate_scores"])
                self.assertEqual(untouched["dataset_hashes_by_noise_seed"],
                                 altered["dataset_hashes_by_noise_seed"])
                for a, b in zip(untouched["runs"], altered["runs"]):
                    # Exclude timing; timings are never identical between executions.
                    self.assertEqual(a["validation_mse"], b["validation_mse"])
                    self.assertEqual(a["theta_final"], b["theta_final"])
                    self.assertEqual(a["termination_reason"], b["termination_reason"])


if __name__ == "__main__":
    unittest.main()
