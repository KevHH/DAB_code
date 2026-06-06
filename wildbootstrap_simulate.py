import argparse
import os
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Wild bootstrap simulation runner")
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        choices=[
            "GaussianShift",
            "GaussianScale",
            "RademacherShift",
            "RademacherScale",
            "CentredGammaShift",
            "CentredGammaScale",
        ],
    )
    parser.add_argument("--compose-transform-names", type=str, nargs="+", required=True)
    parser.add_argument("--average-transform-names", type=str, nargs="+", default=[])
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--testparam", type=float, default=0.0)

    parser.add_argument("--n", type=int, nargs="+", default=[5, 30])
    parser.add_argument("--num-transform", type=int, nargs="+", default=[100])
    parser.add_argument("--num-average-transforms", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-sim", type=int, default=500)
    parser.add_argument("--alpha", type=float, nargs="+", default=list(np.arange(0.00, 0.31, 0.01)))
    parser.add_argument("--quantile-method", type=str, default="linear")
    parser.add_argument("--one-sided", type=str, choices=["upper", "lower"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main():
    # Ensure relative paths in settings/wild_bootstrap.py resolve even when invoked outside the repo root.
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    from settings.wild_bootstrap import simulate_rbf_kernel
    from utils.dabstats import DABConfig

    args = parse_args()

    cfg = DABConfig(
        n=args.n,
        d=args.d,
        num_transform=args.num_transform,
        compose_transform_names=args.compose_transform_names,
        average_transform_names=args.average_transform_names,
        num_average_transforms=args.num_average_transforms,
        label=args.label,
        seed=args.seed,
        num_sim=args.num_sim,
        alpha=args.alpha,
        quantile_method=args.quantile_method,
        one_sided=args.one_sided,
        testparam=args.testparam,
        batch_size=args.batch_size,
    )

    simulate_rbf_kernel(cfg, args.setting)


if __name__ == "__main__":
    main()
