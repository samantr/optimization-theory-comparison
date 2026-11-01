"""Second pilot: test Saman's half-gap rate recovery on two NEW noise seeds.

No test labels, no frozen evaluation seeds, and no modification of the first
pilot's CSV. This is descriptive exploratory development, not confirmation.

Preview: python -m scripts.run_recovering_gd_pilot
Execute: python -m scripts.run_recovering_gd_pilot --execute
"""

import argparse

from scripts.run_adaptive_gd_pilot import execute_pilot


PILOT_SEEDS = {"exponential_decay": 3200, "logistic": 3300}
VARIANTS = ("fixed_gd", "gradient_reversal", "backtracking", "recovering_backtracking")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        execute_pilot(pilot_seeds=PILOT_SEEDS, variants=VARIANTS,
                      filename="recovering_gd_pilot.csv")
    else:
        print("Dry-run: 2 models x 1 NEW seed x 5 original starts x 4 methods = 40 runs.")
        print("New pilot seeds: Model 1=3200, Model 2=3300. No held-out test scoring.")
        print("Comparing fixed GD, reversal, backtracking, and recovering backtracking.")
        print("Add --execute to run; output: results/recovering_gd_pilot.csv")


if __name__ == "__main__":
    main()
