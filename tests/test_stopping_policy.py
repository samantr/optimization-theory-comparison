"""Freeze and validate the approved stopping caps without running the grid."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from opttheory.experiments import optimizer_stopping_kwargs
from opttheory.optimizers import gradient_descent, newton, safeguarded_newton


PLAN = Path(__file__).resolve().parents[1] / "config" / "experiment_plan.json"


class StoppingPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads(PLAN.read_text(encoding="utf-8"))

    def test_approved_shared_tolerance_and_budgets(self):
        self.assertEqual(self.plan["optimizer_stopping"]["status"],
                         "approved_for_both_models_and_calibration")
        for method, cap in (("gradient_descent", 10_000), ("newton", 100),
                            ("safeguarded_newton", 100)):
            with self.subTest(method=method):
                kwargs = optimizer_stopping_kwargs(self.plan, method)
                self.assertEqual(kwargs, {"max_iterations": cap,
                                          "gradient_tolerance": 0.0001})

    def test_methods_stop_early_with_approved_policy(self):
        # At x=0, a+c=5 perfectly fits y=5, so all methods must stop at
        # iteration zero, despite their higher approved iteration caps.
        theta, x, y = [2.0, -0.5, 3.0], [0.0], [5.0]
        for name, solver in (("gradient_descent", gradient_descent),
                             ("newton", newton),
                             ("safeguarded_newton", safeguarded_newton)):
            with self.subTest(method=name):
                kwargs = optimizer_stopping_kwargs(self.plan, name)
                if name == "gradient_descent":
                    kwargs["learning_rate"] = 0.01  # arbitrary here; no step is made
                result = solver(theta, x, y, **kwargs)
                self.assertEqual(result.reason, "gradient_tolerance")
                self.assertEqual(result.iterations, 0)

    def test_unregistered_method_and_missing_policy_rejected(self):
        with self.assertRaisesRegex(ValueError, "registered"):
            optimizer_stopping_kwargs(self.plan, "made_up_solver")
        changed = deepcopy(self.plan)
        del changed["optimizer_stopping"]["max_iterations_by_method"]["newton"]
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            optimizer_stopping_kwargs(changed, "gradient_descent")

    def test_invalid_tolerance_or_iteration_caps_rejected(self):
        for bad_tol in (0, -1, float("inf")):
            with self.subTest(tolerance=bad_tol):
                changed = deepcopy(self.plan)
                changed["optimizer_stopping"]["gradient_norm_tolerance"] = bad_tol
                with self.assertRaisesRegex(ValueError, "positive"):
                    optimizer_stopping_kwargs(changed, "newton")
        for bad_cap in (0, -1, 1.5, True):
            with self.subTest(budget=bad_cap):
                changed = deepcopy(self.plan)
                changed["optimizer_stopping"]["max_iterations_by_method"][
                    "gradient_descent"] = bad_cap
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    optimizer_stopping_kwargs(changed, "gradient_descent")


if __name__ == "__main__":
    unittest.main()
