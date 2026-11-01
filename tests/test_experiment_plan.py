"""Validate approved study breadth and Models 1/2 pre-registered starts."""

import json
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from opttheory.experiments import (
    build_gd_calibration_plan,
    build_model1_plan,
    build_model2_plan,
)
from opttheory.datasets.synthetic import generate_model1


PLAN = Path(__file__).resolve().parents[1] / "config" / "experiment_plan.json"
MODEL1 = PLAN.parent / "model1.json"


class ExperimentPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads(PLAN.read_text(encoding="utf-8"))

    def test_approved_condition_counts_and_method_groups(self):
        p = self.plan
        self.assertEqual(p["model_families"], ["exponential_decay", "logistic"])
        self.assertEqual(p["optional_future_models"], ["gaussian_peak", "oscillatory_trend"])
        self.assertEqual(p["dataset_seeds_per_model"], 10)
        self.assertEqual(p["starting_points_per_dataset"], 5)
        self.assertEqual(2 * 10 * 5, 100)
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
        self.assertEqual(len(set(flattened)), 20)

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

    def test_model2_starting_vectors_match_approval(self):
        p = self.plan
        self.assertEqual(p["model2_parameter_order"], ["L", "k", "x0"])
        starts = p["model2_starting_points"]
        self.assertEqual([s["id"] for s in starts], ["S1", "S2", "S3", "S4", "S5"])
        vectors = [tuple(s["theta"]) for s in starts]
        self.assertEqual(vectors, [
            (9.0, 1.0, 2.0),
            (6.0, 0.5, 1.0),
            (14.0, 3.0, 3.0),
            (10.0, -1.0, 2.0),
            (7.0, 0.15, 0.0),
        ])
        self.assertEqual(len(set(vectors)), 5)
        self.assertTrue(np.all(np.isfinite(np.asarray(vectors, dtype=float))))

    def test_model2_manifest_is_matched_and_keeps_negative_k_start(self):
        runs = build_model2_plan(self.plan)
        self.assertEqual(len(runs), 10 * 5 * 3)
        self.assertEqual(runs, build_model2_plan(self.plan))
        paired = {}
        for run in runs:
            paired.setdefault(run.pairing_key, []).append(run)
        self.assertEqual(len(paired), 50)
        self.assertEqual({run.model for run in runs}, {"logistic"})
        self.assertEqual({run.noise_seed for run in runs}, set(range(1100, 1110)))
        for group in paired.values():
            self.assertEqual([r.method for r in group],
                             ["gradient_descent", "newton", "safeguarded_newton"])
            self.assertEqual(len({r.theta0 for r in group}), 1)
            self.assertEqual(len({r.noise_seed for r in group}), 1)
            self.assertEqual({r.split_seed for r in group}, {2026})
        self.assertEqual({run.theta0 for run in runs if run.start_id == "S4"},
                         {(10.0, -1.0, 2.0)})

    def test_model2_invalid_start_is_rejected_before_experiments(self):
        changed = deepcopy(self.plan)
        changed["model2_starting_points"][0]["theta"] = [9.0, float("nan"), 2.0]
        with self.assertRaisesRegex(ValueError, "Starting vectors"):
            build_model2_plan(changed)

    def test_approved_calibration_grid_and_seed_separation(self):
        tuning = self.plan["gradient_descent_learning_rate_tuning"]
        self.assertEqual(tuning["candidate_learning_rates"],
                         [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3])
        self.assertEqual(tuning["calibration_datasets_per_model"], 3)
        self.assertEqual(tuning["calibration_noise_seeds_by_model"], {
            "exponential_decay": [2000, 2001, 2002],
            "logistic": [2100, 2101, 2102],
        })
        self.assertEqual(tuning["split_seed"], 2026)
        evaluation = {
            value for group in self.plan["seed_schedule"]["noise_seeds_by_model"].values()
            for value in group
        }
        calibration = {
            value for group in tuning["calibration_noise_seeds_by_model"].values()
            for value in group
        }
        self.assertFalse(evaluation.intersection(calibration))
        self.assertEqual(len(calibration), 6)

    def test_gd_calibration_manifest_counts_and_approved_starts(self):
        models = self.plan["model_families"]
        for index, model in enumerate(models, start=1):
            with self.subTest(model=model):
                trials = build_gd_calibration_plan(self.plan, model)
                self.assertEqual(len(trials), 3 * 5 * 8)
                self.assertEqual(trials, build_gd_calibration_plan(self.plan, model))
                self.assertEqual({t.dataset_id for t in trials}, {"C01", "C02", "C03"})
                self.assertEqual({t.start_id for t in trials}, {"S1", "S2", "S3", "S4", "S5"})
                self.assertEqual({t.split_seed for t in trials}, {2026})
                self.assertEqual({t.method for t in trials}, {"gradient_descent"})
                self.assertEqual({t.model for t in trials}, {model})
                rates = self.plan["gradient_descent_learning_rate_tuning"]["candidate_learning_rates"]
                for dataset_id in ("C01", "C02", "C03"):
                    for start in self.plan[f"model{index}_starting_points"]:
                        group = [t for t in trials if t.dataset_id == dataset_id
                                 and t.start_id == start["id"]]
                        self.assertEqual([t.learning_rate for t in group], rates)
                        self.assertEqual({t.theta0 for t in group}, {tuple(start["theta"])})
                if model == "logistic":
                    self.assertEqual({t.theta0 for t in trials if t.start_id == "S4"},
                                     {(10.0, -1.0, 2.0)})

    def test_calibration_rejects_leakage_and_invalid_rates(self):
        for bad_seed in (1000, 1100):
            with self.subTest(bad_seed=bad_seed):
                changed = deepcopy(self.plan)
                changed["gradient_descent_learning_rate_tuning"][
                    "calibration_noise_seeds_by_model"]["logistic"][0] = bad_seed
                with self.assertRaisesRegex(ValueError, "separate"):
                    build_gd_calibration_plan(changed, "logistic")
        for rate in (0.0, -0.1, float("nan")):
            with self.subTest(rate=rate):
                changed = deepcopy(self.plan)
                changed["gradient_descent_learning_rate_tuning"][
                    "candidate_learning_rates"][0] = rate
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    build_gd_calibration_plan(changed, "exponential_decay")


if __name__ == "__main__":
    unittest.main()
