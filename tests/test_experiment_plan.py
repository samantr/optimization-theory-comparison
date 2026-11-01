"""Validate approved study breadth and Model 1 pre-registered starting points."""

import json
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from opttheory.experiments import build_model1_plan
from opttheory.datasets.synthetic import generate_model1


PLAN = Path(__file__).resolve().parents[1] / "config" / "experiment_plan.json"
MODEL1 = PLAN.parent / "model1.json"


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
        self.assertEqual(p["primary_noise"]["distribution"], "gaussian")
        self.assertEqual(p["primary_noise"]["sigma"], 0.1)
        self.assertEqual(p["additional_noise_levels"], "deferred_robustness_study")
        tuning = p["gradient_descent_learning_rate_tuning"]
        self.assertEqual(tuning["scope"], "one_rate_per_model")
        self.assertEqual(tuning["selection_metric"], "validation_mse")
        self.assertIn("separate_calibration", tuning["selection_data"])
        self.assertIn("all_ten_evaluation_datasets", tuning["reuse_policy"])

    def test_approved_observation_count_and_split(self):
        p = self.plan
        self.assertEqual(p["observations_per_dataset"], 120)
        split = p["split"]
        self.assertEqual(split["assignment"], "seeded_random")
        self.assertAlmostEqual(split["train_fraction"], 0.6)
        self.assertAlmostEqual(split["validation_fraction"], 0.2)
        self.assertAlmostEqual(split["test_fraction"], 0.2)
        self.assertEqual([split["train_count"], split["validation_count"],
                          split["test_count"]], [72, 24, 24])
        self.assertEqual(sum([split["train_count"], split["validation_count"],
                              split["test_count"]]), p["observations_per_dataset"])

    def test_seed_schedule_is_disjoint_across_models(self):
        schedule = self.plan["seed_schedule"]
        self.assertEqual(schedule["split_seed"], 2026)
        self.assertEqual(set(schedule["noise_seeds_by_model"]),
                         set(self.plan["model_families"]))
        by_model = list(schedule["noise_seeds_by_model"].values())
        self.assertTrue(all(len(seeds) == 10 and len(set(seeds)) == 10
                            for seeds in by_model))
        flattened = [seed for seeds in by_model for seed in seeds]
        self.assertEqual(len(set(flattened)), 40)

    def test_fixed_split_and_different_noise_are_reproducible(self):
        prototype = json.loads(MODEL1.read_text(encoding="utf-8"))
        def sample(noise_seed):
            config = deepcopy(prototype)
            config["noise"]["seed"] = noise_seed
            config["split"]["seed"] = self.plan["seed_schedule"]["split_seed"]
            return generate_model1(config)

        a = sample(1000)
        b = sample(1001)
        again = sample(1000)
        np.testing.assert_array_equal(a.split, b.split)
        np.testing.assert_array_equal(a.noise, again.noise)
        self.assertFalse(np.array_equal(a.noise, b.noise))
        self.assertEqual(a.split_counts(), {"train": 72, "validation": 24, "test": 24})

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

    def test_manifest_pairs_all_three_methods_on_identical_conditions(self):
        runs = build_model1_plan(self.plan)
        self.assertEqual(len(runs), 10 * 5 * 3)
        self.assertEqual(runs, build_model1_plan(self.plan))
        paired = {}
        for run in runs:
            paired.setdefault(run.pairing_key, []).append(run)
        self.assertEqual(len(paired), 50)
        self.assertEqual({key[1] for key in paired},
                         {f"D{i:02d}" for i in range(1, 11)})
        for group in paired.values():
            self.assertEqual([r.method for r in group],
                             ["gradient_descent", "newton", "safeguarded_newton"])
            self.assertEqual(len({r.theta0 for r in group}), 1)
            self.assertEqual(len({r.dataset_id for r in group}), 1)
            self.assertEqual(len({r.noise_seed for r in group}), 1)
            self.assertEqual({r.split_seed for r in group}, {2026})


if __name__ == "__main__":
    unittest.main()
