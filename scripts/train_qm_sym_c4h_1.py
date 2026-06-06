#!/usr/bin/env python
"""Train the amptorch SingleNN/GMP model on QM-sym C4h_1 ASE Atoms."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.calculators.singlepoint import SinglePointCalculator
from amptorch.trainer import AtomsTrainer
from skorch.callbacks import Callback


DEFAULT_ASE_PATH = Path("data/qm_sym_c4h_1/qm_sym_c4h_1_ase_u0_ha.pkl")
DEFAULT_RUN_DIR = Path("logs/qm_sym_c4h_1_paper")
DEFAULT_IDENTIFIER = "qm_sym_c4h_1_gmp30_snn128_64_64"
TARGET_NAME = "sum_electronic_zero_point_energy_ha"
SIGMAS = np.linspace(0.02, 2.0, 30, endpoint=True).tolist()
GMP_PARAMS = {"MCSHs": {"orders": [0, 1, 2], "sigmas": SIGMAS}, "cutoff": 15.0}
HIDDEN_LAYERS = [128, 64, 64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-path", default=str(DEFAULT_ASE_PATH), help="Pickle from qm_sym_to_ase.py.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="Directory for checkpoints/logs.")
    parser.add_argument("--identifier", default=DEFAULT_IDENTIFIER, help="amptorch run identifier suffix.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Optional number of structures to train on after --limit is applied. Defaults to all available.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional molecule limit for smoke tests.")
    parser.add_argument("--epochs", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument(
        "--csv-log",
        default=None,
        help="Per-epoch CSV log path. Default: training_log.csv inside the checkpoint directory.",
    )
    parser.add_argument("--no-save-fps", action="store_true", help="Disable fingerprint caching.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --device cuda, but torch.cuda.is_available() is false.")
    return device_arg


def load_images(path: Path) -> list:
    with path.open("rb") as handle:
        images = pickle.load(handle)
    if not isinstance(images, list):
        raise TypeError(f"Expected {path} to contain a list, got {type(images)!r}.")
    return images


def select_train_images(
    images: list,
    train_size: int | None,
    limit: int | None,
    seed: int,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive when provided.")
    if train_size is not None and train_size <= 0:
        raise ValueError("--train-size must be positive when provided.")

    available_count = len(images) if limit is None else min(limit, len(images))
    available_indices = np.arange(available_count, dtype=np.int64)
    actual_train_size = available_count if train_size is None else train_size
    if actual_train_size > available_count:
        raise ValueError(f"--train-size {actual_train_size} exceeds available size {available_count}.")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(available_indices)
    train_indices = shuffled[:actual_train_size]
    holdout_indices = shuffled[actual_train_size:]
    train_list = [images[int(idx)] for idx in train_indices]
    return train_list, train_indices, holdout_indices, available_indices


def count_elements(images: list, elements: list[str]) -> np.ndarray:
    element_to_col = {element: idx for idx, element in enumerate(elements)}
    counts = np.zeros((len(images), len(elements)), dtype=np.float64)
    for row, atoms in enumerate(images):
        for symbol in atoms.get_chemical_symbols():
            counts[row, element_to_col[symbol]] += 1.0
    return counts


def fit_element_bias(images: list) -> tuple[dict, np.ndarray]:
    elements = sorted({symbol for atoms in images for symbol in atoms.get_chemical_symbols()})
    counts = count_elements(images, elements)
    energies = np.array([atoms.get_potential_energy() for atoms in images], dtype=np.float64)
    coeffs, residuals, rank, singular_values = np.linalg.lstsq(counts, energies, rcond=None)
    corrections = counts @ coeffs
    corrected_targets = energies - corrections
    metadata = {
        "type": "linear_per_element_counts",
        "fit_target": TARGET_NAME,
        "target_units": "Hartree",
        "include_intercept": False,
        "elements": elements,
        "coefficients_ha": {element: float(coeffs[idx]) for idx, element in enumerate(elements)},
        "rank": int(rank),
        "singular_values": [float(value) for value in singular_values],
        "lstsq_residuals": [float(value) for value in residuals],
        "train_corrected_target_mean_ha": float(np.mean(corrected_targets)),
        "train_corrected_target_std_ha": float(np.std(corrected_targets)),
    }
    return metadata, corrections


def apply_element_bias(images: list, corrections: np.ndarray) -> list:
    corrected_images = []
    for atoms, correction in zip(images, corrections):
        corrected = atoms.copy()
        original_energy = float(atoms.get_potential_energy())
        corrected_energy = original_energy - float(correction)
        corrected.calc = SinglePointCalculator(corrected, energy=corrected_energy)
        corrected.info["original_energy_ha"] = original_energy
        corrected.info["element_bias_correction_ha"] = float(correction)
        corrected.info["training_target_ha"] = corrected_energy
        corrected_images.append(corrected)
    return corrected_images


def build_config(args: argparse.Namespace, train_list: list, run_dir: Path, device: str) -> dict:
    gpus = 1 if device == "cuda" else 0
    return {
        "model": {
            "name": "singlenn",
            "get_forces": False,
            "hidden_layers": HIDDEN_LAYERS,
            "activation": torch.nn.GELU,
            "batchnorm": True,
        },
        "optim": {
            "device": device,
            "gpus": gpus,
            "force_coefficient": 0.0,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "scheduler": {
                "policy": torch.optim.lr_scheduler.StepLR,
                "params": {"step_size": 2000, "gamma": 0.5},
            },
        },
        "dataset": {
            "raw_data": train_list,
            "val_split": args.val_split,
            "fp_scheme": "gmpordernorm",
            "fp_params": deepcopy(GMP_PARAMS),
            "save_fps": not args.no_save_fps,
        },
        "cmd": {
            "debug": False,
            "run_dir": str(run_dir),
            "seed": args.seed,
            "identifier": args.identifier,
            "verbose": True,
            "logger": False,
        },
    }


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf8") as handle:
        json.dump(payload, handle, indent=2)


class CsvEpochLogger(Callback):
    """Append one CSV row per epoch and echo the same summary to stdout."""

    def __init__(self, path: Path, echo_stdout: bool = True):
        self.path = Path(path)
        self.echo_stdout = echo_stdout
        self.rows: list[dict[str, Any]] = []
        self.fieldnames: list[str] = []

    def on_train_begin(self, net, **kwargs):
        self.rows = []
        self.fieldnames = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        print(f"CSV epoch log: {self.path}", flush=True)

    def on_epoch_end(self, net, **kwargs):
        row = self._clean_epoch_row(dict(net.history[-1]))
        self.rows.append(row)

        new_fields = [key for key in row if key not in self.fieldnames]
        if new_fields:
            self.fieldnames.extend(new_fields)
            self._rewrite()
        else:
            self._append(row)

        if self.echo_stdout:
            print(self._format_stdout(row), flush=True)

    def _clean_epoch_row(self, row: dict[str, Any]) -> dict[str, Any]:
        cleaned = {}
        for key, value in row.items():
            if key == "batches":
                continue
            cleaned[key] = self._csv_value(value)
        return cleaned

    @staticmethod
    def _csv_value(value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return json.dumps(value)

    def _append(self, row: dict[str, Any]) -> None:
        with self.path.open("a", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow({key: row.get(key, "") for key in self.fieldnames})

    def _rewrite(self) -> None:
        with self.path.open("w", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({key: row.get(key, "") for key in self.fieldnames})

    def _format_stdout(self, row: dict[str, Any]) -> str:
        keys = [
            "epoch",
            "train_loss",
            "valid_loss",
            "train_energy_mae",
            "val_energy_mae",
            "event_lr",
            "dur",
        ]
        parts = [f"{key}={row[key]}" for key in keys if key in row]
        return "csv_epoch_log " + " ".join(parts)


def main() -> None:
    args = parse_args()
    ase_path = Path(args.ase_path)
    run_dir = Path(args.run_dir).resolve()
    csv_log_path = Path(args.csv_log).resolve() if args.csv_log else None
    run_dir.mkdir(parents=True, exist_ok=True)

    images = load_images(ase_path)
    train_images, train_indices, holdout_indices, available_indices = select_train_images(
        images, args.train_size, args.limit, args.seed
    )

    bias_metadata, corrections = fit_element_bias(train_images)
    corrected_train_list = apply_element_bias(train_images, corrections)

    split_path = run_dir / "qm_sym_c4h_1_split_indices.npz"
    bias_path = run_dir / "qm_sym_c4h_1_element_bias.json"
    config_path = run_dir / "qm_sym_c4h_1_train_config.json"
    np.savez(
        split_path,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
        available_indices=available_indices,
        seed=np.array(args.seed),
        limit=np.array(-1 if args.limit is None else args.limit),
    )
    write_json(bias_path, bias_metadata)

    device = resolve_device(args.device)
    config = build_config(args, corrected_train_list, run_dir, device)
    config_metadata = {
        "ase_path": str(ase_path),
        "num_loaded": len(images),
        "limit": args.limit,
        "num_available": len(available_indices),
        "num_train": len(corrected_train_list),
        "num_holdout": len(holdout_indices),
        "device": device,
        "gpus": config["optim"]["gpus"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "scheduler": {"policy": "StepLR", "step_size": 2000, "gamma": 0.5},
        "val_split": args.val_split,
        "identifier": args.identifier,
        "model": {
            "name": "singlenn",
            "hidden_layers": HIDDEN_LAYERS,
            "activation": "GELU",
            "batchnorm": True,
            "get_forces": False,
        },
        "fingerprints": deepcopy(GMP_PARAMS),
        "save_fps": not args.no_save_fps,
        "target_before_bias": TARGET_NAME,
        "training_target": f"{TARGET_NAME} minus fitted per-element bias",
        "split_indices_path": str(split_path),
        "element_bias_path": str(bias_path),
    }
    write_json(config_path, config_metadata)

    print(f"Loaded {len(images)} molecules from {ase_path}")
    if args.limit is not None:
        print(f"Limited run to the first {len(available_indices)} molecules for smoke testing")
    print(f"Training on {len(corrected_train_list)} molecules; holdout size {len(holdout_indices)}")
    print(f"Fitted per-element bias on training split and wrote {bias_path}")
    print(f"Device request resolved to {device}; amptorch gpus={config['optim']['gpus']}")
    print(f"Run directory: {run_dir}")

    trainer = AtomsTrainer(config)
    trainer.load()

    if csv_log_path is None:
        csv_log_path = Path(trainer.cp_dir) / "training_log.csv"
    trainer.net.callbacks.append(("csv_epoch_logger", CsvEpochLogger(csv_log_path)))

    config_metadata["checkpoint_dir"] = trainer.cp_dir
    config_metadata["csv_log_path"] = str(csv_log_path)
    write_json(config_path, config_metadata)

    stime = time.time()
    trainer.net.fit(trainer.train_dataset, None)
    elapsed_time = time.time() - stime
    print(f"Training completed in {elapsed_time}s")

    write_json(config_path, config_metadata)
    print(f"Training complete. Checkpoint directory: {trainer.cp_dir}")
    print(f"CSV epoch log: {csv_log_path}")


if __name__ == "__main__":
    main()
