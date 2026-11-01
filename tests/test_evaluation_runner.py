"""Pilot-only tests: NEVER use the ten registered final evaluation seeds."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from opttheory.datasets.synthetic import generate_model1, generate_model2
from opttheory.experiments.analysis import summarize_paired_primary
from opttheory.experiments.evaluation import (
    METHODS, _curvature, balanced_method_order,
    evaluate_condition, validate_evaluation_protocol,
)
from opttheory.experiments.plan import PlannedRun, build_model1_plan
from scripts.run_comparison import main


ROOT = Path(__file__).resolve().parents[1]


class EvaluationRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((ROOT / "config/experiment_plan.json").read_text())
        cls.frozen = json.loads((ROOT / "config/selected_gd_rates.json").read_text())
        cls.protocol = json.loads((ROOT / "config/evaluation_protocol.json").read_text())

    def _pilot(self, model="exponential_decay", method="gradient_descent"):
        number = 1 if model == "exponential_decay" else 2
        cfg = json.loads((ROOT / f"config/model{number}.json").read_text())
        cfg["noise"]["seed"] = 42  # pilot, not any registered final seed
        data = (generate_model1 if number == 1 else generate_model2)(cfg)
        theta = [3.0, -0.5, 1.0] if number == 1 else [9.0, 1.0, 2.0]
        trial = PlannedRun(model, "PILOT", 42, 2026, "S1", tuple(theta), method)
        plan = deepcopy(self.plan)
        plan["optimizer_stopping"]["max_iterations_by_method"][method] = 2
        return data, trial, plan

    def test_preview_is_safe_and_reports_all_1500_timed_invocations(self):
        manifest = validate_evaluation_protocol(self.plan, self.frozen, self.protocol)
        self.assertEqual(manifest["exponential_decay"]["frozen_gd_learning_rate"], 0.1)
        self.assertEqual(manifest["logistic"]["frozen_gd_learning_rate"], 0.03)
        self.assertEqual(sum(x["timed_invocations"] for x in manifest.values()), 1500)
        with patch("scripts.run_comparison.run_full_model_evaluation",
                   side_effect=AssertionError("final evaluation must not run")):
            with redirect_stdout(StringIO()) as output:
                main([])
        self.assertIn("PREVIEW ONLY", output.getvalue())

    def test_method_order_is_counterbalanced_without_changing_pairings(self):
        original = build_model1_plan(self.plan)
        balanced = balanced_method_order(original)
        self.assertEqual(set(original), set(balanced))
        self.assertEqual(len(balanced), 150)
        first_methods = [balanced[index].method for index in range(0, 150, 3)]
        self.assertEqual(first_methods.count("gradient_descent"), 25)
        self.assertEqual(first_methods.count("newton"), 25)
        self.assertTrue(all(balanced[index + 2].method == "safeguarded_newton"
                            for index in range(0, 150, 3)))

    def test_mutated_frozen_manifest_and_rate_are_rejected_before_experiment(self):
        plan = deepcopy(self.plan)
        plan["seed_schedule"]["split_seed"] += 1
        with self.assertRaisesRegex(ValueError, "changed after"):
            validate_evaluation_protocol(plan, self.frozen, self.protocol)
        frozen = deepcopy(self.frozen)
        frozen["learning_rates_by_model"]["logistic"] = 0.123456
        with self.assertRaisesRegex(ValueError, "calibration candidates"):
            validate_evaluation_protocol(self.plan, frozen, self.protocol)
        protocol = deepcopy(self.protocol)
        protocol["timing"]["repetitions_per_condition"] = 1
        with self.assertRaisesRegex(ValueError, "five"):
            validate_evaluation_protocol(self.plan, self.frozen, protocol)

    def test_five_independent_pilot_solver_calls_and_posthoc_scores(self):
        for model, method in (("exponential_decay", "gradient_descent"),
                              ("logistic", "newton"),
                              ("exponential_decay", "safeguarded_newton")):
            with self.subTest(model=model, method=method):
                data, trial, plan = self._pilot(model, method)
                original = METHODS[method]
                seen = []

                def checked_solver(theta0, x, y, **kwargs):
                    seen.append((tuple(theta0), x.copy(), y.copy()))
                    return original(theta0, x, y, **kwargs)

                with patch.dict(METHODS, {method: checked_solver}):
                    row, timings = evaluate_condition(data, trial, plan,
                                                      self.frozen, self.protocol)
                self.assertEqual(len(timings), 5)
                self.assertEqual([r["repetition"] for r in timings], [1, 2, 3, 4, 5])
                self.assertTrue(all(t["elapsed_seconds"] >= 0 for t in timings))
                self.assertEqual(row["repetitions"], 5)
                self.assertTrue(all(t[0] == trial.theta0 for t in seen))
                for theta0, x, y in seen:
                    np.testing.assert_array_equal(x, data.x[data.split == "train"])
                    np.testing.assert_array_equal(y, data.y_observed[data.split == "train"])
                self.assertIsNotNone(row["train_mse"])
                self.assertIsNotNone(row["validation_mse"])
                self.assertIsNotNone(row["test_mse"])
                self.assertEqual(row["gd_learning_rate"],
                                 self.frozen["learning_rates_by_model"][model]
                                 if method == "gradient_descent" else None)

    def test_test_target_mutation_affects_only_posthoc_test_mse(self):
        data, trial, plan = self._pilot()
        original, _ = evaluate_condition(data, trial, plan, self.frozen, self.protocol)
        changed_y = data.y_observed.copy()
        changed_y[data.split == "test"] += 1000
        changed = replace(data, y_observed=changed_y)
        altered, _ = evaluate_condition(changed, trial, plan, self.frozen, self.protocol)
        self.assertEqual(original["final_theta"], altered["final_theta"])
        self.assertEqual(original["train_mse"], altered["train_mse"])
        self.assertEqual(original["validation_mse"], altered["validation_mse"])
        self.assertNotEqual(original["test_mse"], altered["test_mse"])

    def test_expected_numerical_failure_keeps_all_timing_rows(self):
        data, trial, plan = self._pilot()
        with patch.dict(METHODS, {trial.method: lambda *a, **k: (_ for _ in ()).throw(
            FloatingPointError("pilot injected failure"))}):
            row, timings = evaluate_condition(data, trial, plan,
                                              self.frozen, self.protocol)
        self.assertEqual(row["termination_reason"], "numerical_failure")
        self.assertFalse(row["stationary_at_tolerance"])
        self.assertIsNone(row["final_theta"])
        self.assertEqual(len(timings), 5)
        self.assertTrue(all(r["termination_reason"] == "numerical_failure" for r in timings))

    def test_inconsistent_repeat_raises_and_cannot_be_silently_medianed(self):
        data, trial, plan = self._pilot()
        original = METHODS[trial.method]
        counter = [0]

        def changing_solver(*args, **kwargs):
            counter[0] += 1
            result = original(*args, **kwargs)
            return replace(result, theta=result.theta + counter[0] * 1e-3)

        with patch.dict(METHODS, {trial.method: changing_solver}):
            with self.assertRaisesRegex(RuntimeError, "deterministic"):
                evaluate_condition(data, trial, plan, self.frozen, self.protocol)

    def test_hessian_curvature_labels_are_diagnostic_not_success_guarantees(self):
        theta, x, y = np.zeros(3), np.array([0.0]), np.array([1.0])
        for vals, expected in [
            ([2.0, 3.0, 4.0], "strict_local_minimum"),
            ([-2.0, -3.0, -4.0], "strict_local_maximum"),
            ([2.0, -3.0, 4.0], "indefinite_saddle"),
            ([2.0, 0.0, 4.0], "inconclusive_near_zero_curvature"),
        ]:
            with self.subTest(vals=vals):
                result = _curvature(lambda *args: np.diag(vals), theta, x, y, 1e-8)
                self.assertEqual(result["curvature"], expected)

    def test_dataset_aware_aggregation_uses_two_independent_units_not_four(self):
        rows = []
        for dataset, times in (("P01", [0.1, 0.3]), ("P02", [0.2, 0.4])):
            for index, diff in enumerate(times, 1):
                for method, runtime in (("gradient_descent", 1.0),
                                        ("newton", 1.0 + diff)):
                    rows.append({
                        "model": "exponential_decay", "dataset_id": dataset,
                        "start_id": f"S{index}", "noise_seed": int(dataset[-1]),
                        "split_seed": 2026,
                        "theta_initial": [index, 1.0, 1.0], "method": method,
                        "median_optimizer_seconds": runtime,
                        "stationary_at_tolerance": (method == "gradient_descent"),
                    })
        summary = summarize_paired_primary(rows, expected_starts=2, expected_datasets=2)
        result = summary["exponential_decay"]
        self.assertEqual(result["independent_dataset_count"], 2)
        self.assertEqual(result["paired_condition_count"], 4)
        self.assertAlmostEqual(result["dataset_mean_newton_minus_gd_seconds"], 0.25)
        self.assertEqual(result["gd_stationary_fraction"], 1.0)
        self.assertEqual(result["newton_stationary_fraction"], 0.0)
        with self.assertRaisesRegex(ValueError, "missing"):
            summarize_paired_primary(rows[:-1], expected_starts=2, expected_datasets=2)


if __name__ == "__main__":
    unittest.main()
