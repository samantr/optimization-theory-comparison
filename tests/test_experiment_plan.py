"""Validate approved study breadth and Model 1 pre-registered starting points."""

import json
from pathlib import Path
import unittest

import numpy as np


PLAN = Path(__file__).resolve().parents[1] / "config" / "experiment_plan.json"


class ExperimentPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads(PLAN.read_text(encoding="utf-8"))

    def test_approved_condition_counts_and_method_groups(self):
        p = self.plan
        self.assertEqual(len(p["model_families"]), 4)
        self.assertEqual(p["dataset_seeds_per_model"], 10)
        self.assertEqual(p["starting_points_per_dataset"], 5)
        self.assertEqual(4 * 10 * 5, 200)
        self.assertEqual(p["primary_methods"], ["gradient_descent", "newton"])
        self.assertEqual(p["exploratory_methods"], ["safeguarded_newton"])

    def test_model1_starts_are_fixed_distinct_and_finite(self):
        p = self.plan
        self.assertEqual(p["model1_parameter_order"], ["a", "b", "c"])
        starts = p["model1_starting_points"]
        self.assertEqual(len(starts), p["starting_points_per_dataset"])
        self.assertEqual([s["id"] for s in starts], ["S1", "S2", "S3", "S4", "S5"])
        vectors = [tuple(s["theta"]) for s in starts]
        self.assertEqual(len(set(vectors)), 5)
        self.assertTrue(all(len(v) == 3 for v in vectors))
        self.assertTrue(np.all(np.isfinite(np.asarray(vectors, dtype=float))))
        self.assertEqual(vectors, [
            (3.0, -0.5, 1.0),
            (1.0, -1.2, 0.0),
            (6.0, -0.2, 2.0),
            (-1.0, -0.5, 3.0),
            (2.0, 0.4, 0.0),
        ])


if __name__ == "__main__":
    unittest.main()
