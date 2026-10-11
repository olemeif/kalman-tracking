import argparse

from Train_1D import Train_1D, monte_carlo
from Train_1D_deceleration import Train_1D_deceleration
from Train_1D_deceleration import monte_carlo as monte_carlo_deceleration

# scenario -> {mode: function}. A new scenario or mode is one more entry here.
SCENARIOS = {
    "train": {"run": Train_1D.main, "mc": monte_carlo.main},
    "decel": {"run": Train_1D_deceleration.main, "mc": monte_carlo_deceleration.main},
}

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a Kalman filter scenario: a single run with plots, or a Monte Carlo evaluation.",
        epilog="examples:  python main.py train mc    |    python main.py decel    |    python main.py --all",
    )
    parser.add_argument("scenario", nargs="?", choices=SCENARIOS,
                        help="train: Stage 1, constant speed (2nd order). "
                             "decel: Stage 2, braking train (3rd order).")
    parser.add_argument("mode", nargs="?", choices=["run", "mc"], default="run",
                        help="run: one noise realization with plots (default). "
                             "mc: Monte Carlo evaluation (RMSE, ANEES).")
    parser.add_argument("--all", action="store_true", help="run every scenario in both modes")
    args = parser.parse_args(argv)
    if not args.all and args.scenario is None:
        parser.error("choose a scenario (or --all)")
    return args

if __name__ == "__main__":
    args = parse_args()
    jobs = [(s, m) for s in SCENARIOS for m in ("run", "mc")] if args.all else [(args.scenario, args.mode)]
    for scenario, mode in jobs:
        print(f"=== {scenario} {mode} ===")
        SCENARIOS[scenario][mode]()
