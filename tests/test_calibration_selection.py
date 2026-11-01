"""Synthetic unit fixtures for calibration-selection policy (NO actual GD runs)."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from opttheory.experiments import (
    CalibrationResult,
    build_gd_calibration_plan,
    select_gd_learning_rate,
)


PLAN = Path(__file__).resolve().parents[1] / "config" / "experiment_plan.json"


class CalibrationSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(PLAN.read_text(encoding="utf-8"))

    def _example_results(self, model):
        """Construct hypothetical A-vs-B outcomes to check the rule."""
        trials = build_gd_calibration_plan(self.config, model)
        output = []
        for trial in trials:
            if trial.learning_rate == 0.03:
                # B: reliable, MSE 0.007, one cap-exhausted finite run.
                reason = "max_iterations" if trial.dataset_id == "C01" and trial.start_id == "S1" else "gradient_tolerance"
                loss = 0.007
            elif trial.learning_rate == 0.1:
                # A: usually better MSE, but fails on one matched start.
                reason = "numerical_failure" if trial.dataset_id == "C01" and trial.start_id == "S1" else "gradient_tolerance"
                loss = None if reason == "numerical_failure" else 0.005
            else:
                reason = "gradient_tolerance"
                loss = 0.02
            output.append(CalibrationResult(trial, reason, loss))
        return output

    def test_stability_first_selects_B_and_preserves_failure_counts(self):
        for model in self.config["model_families"]:
            with self.subTest(model=model):
                results = self._example_results(model)
                rate, scores = select_gd_learning_rate(self.config, model, results)
                self.assertEqual(rate, 0.03)
                self.assertEqual(len(scores), 8)
                by_rate = {score.learning_rate: score for score in scores}
                self.assertEqual(by_rate[0.03].numerical_failures, 0)
                self.assertEqual(by_rate[0.03].budget_exhaustions, 1)
                self.assertEqual(by_rate[0.03].finite_validation_cases, 15)
                self.assertAlmostEqual(by_rate[0.03].mean_validation_mse, 0.007)
                self.assertEqual(by_rate[0.1].numerical_failures, 1)
                self.assertEqual(by_rate[0.1].finite_validation_cases, 14)
                self.assertAlmostEqual(by_rate[0.1].mean_validation_mse, 0.005)

    def test_complete_grid_required_and_no_duplicate_or_forged_trial(self):
        model = "logistic"
        rows = self._example_results(model)
        with self.assertRaisesRegex(ValueError, "complete"):
            select_gd_learning_rate(self.config, model, rows[:-1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select_gd_learning_rate(self.config, model, rows[:-1] + [rows[0]])
        fake = replace(rows[0], trial=replace(rows[0].trial, noise_seed=9999))
        with self.assertRaisesRegex(ValueError, "approved"):
            select_gd_learning_rate(self.config, model, [fake] + rows[1:])

    def test_nonfailed_run_requires_finite_validation_mse(self):
        model = "exponential_decay"
        rows = self._example_results(model)
        with self.assertRaisesRegex(ValueError, "Finite nonnegative"):
            select_gd_learning_rate(self.config, model,
                                    [replace(rows[0], validation_mse=float("nan"))] + rows[1:])
        with self.assertRaisesRegex(ValueError, "Unknown GD"):
            select_gd_learning_rate(self.config, model,
                                    [replace(rows[0], termination_reason="undocumented")] + rows[1:])

    def test_exact_tie_picks_smaller_rate(self):
        trials = build_gd_calibration_plan(self.config, "logistic")
        rows = [CalibrationResult(trial, "gradient_tolerance", 0.01) for trial in trials]
        chosen, _ = select_gd_learning_rate(self.config, "logistic", rows)
        self.assertEqual(chosen, 0.0001)

    def test_all_failed_has_no_valid_selection(self):
        trials = build_gd_calibration_plan(self.config, "logistic")
        rows = [CalibrationResult(trial, "numerical_failure", None) for trial in trials]
        with self.assertRaisesRegex(ValueError, "No finite"):
            select_gd_learning_rate(self.config, "logistic", rows)

    def test_code_refuses_silent_ranking_rule_change(self):
        altered = deepcopy(self.config)
        altered["gradient_descent_learning_rate_tuning"]["selection_policy"]["ranking"] = [
            "best_success_only_mse"
        ]
        with self.assertRaisesRegex(ValueError, "must match"):
            select_gd_learning_rate(altered, "logistic", self._example_results("logistic"))


if __name__ == "__main__":
    unittest.main()
