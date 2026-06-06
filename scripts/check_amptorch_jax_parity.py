#!/usr/bin/env python
"""Check JAX AmpTorch export parity against saved PyTorch references."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.load_amptorch_model import (
    jax_denormalize_energy,
    load_jax_model,
)


def parse_case(raw: str) -> tuple[str, Path, str]:
    parts = raw.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--case must have the form label:export_dir:dataset_name"
        )
    label, export_dir, dataset_name = parts
    return label, Path(export_dir), dataset_name


def max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 and right.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def check_case(
    label: str,
    export_dir: Path,
    dataset_name: str,
    energy_atol: float,
    energy_rtol: float,
    latent_atol: float,
    latent_rtol: float,
) -> dict[str, float]:
    params, target_scaler, _element_bias, jax_energy, jax_latent_fn = load_jax_model(export_dir)

    atom_fps = np.load(export_dir / f"{dataset_name}_atom_fps.npy")
    image_ids = np.load(export_dir / f"{dataset_name}_image_ids.npy")
    bias = np.load(export_dir / f"{dataset_name}_energy_bias_correction_ha.npy")
    ref_norm = np.load(export_dir / f"{dataset_name}_energy_pred_normalized.npy")
    ref_corrected = np.load(export_dir / f"{dataset_name}_energy_pred_corrected_ha.npy")
    ref_physical = np.load(export_dir / f"{dataset_name}_energy_pred_ha.npy")
    ref_latent = np.load(export_dir / f"{dataset_name}_latent.npy")

    num_images = int(ref_norm.shape[0])
    jax_norm = np.asarray(jax_energy(params, atom_fps, image_ids, num_images))
    jax_corrected = np.asarray(jax_denormalize_energy(jax_norm, target_scaler))
    jax_physical = np.asarray(jax_denormalize_energy(jax_norm, target_scaler, bias))
    jax_latent = np.asarray(jax_latent_fn(params, atom_fps, image_ids, num_images))

    results = {
        "num_images": float(num_images),
        "num_atoms": float(atom_fps.shape[0]),
        "energy_norm_max_abs": max_abs_diff(jax_norm, ref_norm),
        "energy_corrected_max_abs_ha": max_abs_diff(jax_corrected, ref_corrected),
        "energy_physical_max_abs_ha": max_abs_diff(jax_physical, ref_physical),
        "latent_max_abs": max_abs_diff(jax_latent, ref_latent),
    }

    energy_ok = np.allclose(jax_physical, ref_physical, atol=energy_atol, rtol=energy_rtol)
    corrected_ok = np.allclose(jax_corrected, ref_corrected, atol=energy_atol, rtol=energy_rtol)
    norm_ok = np.allclose(jax_norm, ref_norm, atol=energy_atol, rtol=energy_rtol)
    latent_ok = np.allclose(jax_latent, ref_latent, atol=latent_atol, rtol=latent_rtol)
    if not (energy_ok and corrected_ok and norm_ok and latent_ok):
        raise AssertionError(
            f"{label}: parity failed "
            f"(norm={results['energy_norm_max_abs']:.3e}, "
            f"corrected={results['energy_corrected_max_abs_ha']:.3e}, "
            f"physical={results['energy_physical_max_abs_ha']:.3e}, "
            f"latent={results['latent_max_abs']:.3e})"
        )

    print(
        f"{label}: ok "
        f"images={num_images} atoms={atom_fps.shape[0]} "
        f"energy_abs={results['energy_physical_max_abs_ha']:.3e} Ha "
        f"latent_abs={results['latent_max_abs']:.3e}"
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        type=parse_case,
        help="Case to check, as label:export_dir:dataset_name. May be repeated.",
    )
    parser.add_argument("--energy-atol", type=float, default=1e-8)
    parser.add_argument("--energy-rtol", type=float, default=1e-10)
    parser.add_argument("--latent-atol", type=float, default=1e-8)
    parser.add_argument("--latent-rtol", type=float, default=1e-10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for label, export_dir, dataset_name in args.case:
        check_case(
            label=label,
            export_dir=export_dir,
            dataset_name=dataset_name,
            energy_atol=args.energy_atol,
            energy_rtol=args.energy_rtol,
            latent_atol=args.latent_atol,
            latent_rtol=args.latent_rtol,
        )


if __name__ == "__main__":
    main()
