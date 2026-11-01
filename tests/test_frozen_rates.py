"""Guard the post-calibration decision against silent preregistration changes."""

import json
from pathlib import Path
import unittest

from opttheory.experiments.run_calibration import canonical_json_hash


ROOT = Path(__file__).resolve().parents[1]


class FrozenCalibrationDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((ROOT / "config/experiment_plan.json").read_text())
        cls.frozen = json.loads((ROOT / "config/selected_gd_rates.json").read_text())

    def test_original_approved_manifest_has_not_been_mutated_after_selection(self):
        self.assertEqual(canonical_json_hash(self.plan),
                         self.frozen["approved_experiment_plan_sha256"])
        self.assertFalse(self.frozen["final_test_or_newton_results_used"])

    def test_selected_rates_were_registered_and_frozen_for_both_models(self):
        rates = self.plan["gradient_descent_learning_rate_tuning"]["candidate_learning_rates"]
        self.assertEqual(set(self.frozen["learning_rates_by_model"]),
                         set(self.plan["model_families"]))
        for model, selected in self.frozen["learning_rates_by_model"].items():
            with self.subTest(model=model):
                self.assertIn(selected, rates)
                audit = self.frozen["models"][model]
                self.assertEqual(audit["calibration_conditions"], 120)
                self.assertEqual(audit["chosen_rate_trials"], 15)
                self.assertEqual(audit["chosen_rate_numerical_failures"], 0)
                self.assertEqual(audit["chosen_rate_budget_exhaustions"], 0)


if __name__ == "__main__":
    unittest.main()
