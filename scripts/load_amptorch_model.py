"""Export an AmpTorch SingleNN checkpoint to JAX-compatible arrays.

The export path intentionally keeps PyTorch/AmpTorch and JAX decoupled:
PyTorch is imported only by :func:`export`, while JAX is imported only by
the JAX loader/forward helpers. This lets the repository use the existing
``amptorch`` conda environment for export and the existing ``DAB`` conda
environment for JAX parity checks.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CHECKPOINT_DIR = Path(
    "logs/qm_sym_c4h_1_paper/checkpoints/"
)
DEFAULT_ASE_PATH = Path("data/qm_sym_c4h_1/qm_sym_c4h_1_ase_u0_ha.pkl")
DEFAULT_SPLIT_INDICES = Path("logs/qm_sym_c4h_1_paper/qm_sym_c4h_1_split_indices.npz")
DEFAULT_ELEMENT_BIAS = Path("logs/qm_sym_c4h_1_paper/qm_sym_c4h_1_element_bias.json")
DEFAULT_OUT_DIR = Path("data/qm_sym_c4h_1_jax")
DEFAULT_LATENT_LAYER = -2


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        json.dump(payload, handle, indent=2)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf8") as handle:
        return json.load(handle)


def _load_ase_images(path: Path) -> list:
    with path.open("rb") as handle:
        images = pickle.load(handle)
    if not isinstance(images, list):
        raise TypeError(f"Expected {path} to contain a list of ASE Atoms, got {type(images)!r}.")
    return images


def _select_indices(
    indices: np.ndarray,
    max_images: int | None,
    sample_seed: int | None,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if max_images is not None and max_images <= 0:
        raise ValueError("--max-images must be positive when provided.")
    if max_images is not None and max_images < len(indices):
        if sample_seed is None:
            return indices[:max_images]
        rng = np.random.default_rng(sample_seed)
        selected = rng.choice(indices, size=max_images, replace=False)
        return np.asarray(selected, dtype=np.int64)
    return indices


def _split_indices(
    num_images: int,
    split_indices_path: Path | None,
    dataset_name: str,
    calib_ratio: float,
) -> dict[str, np.ndarray]:
    if split_indices_path is None:
        return {dataset_name: np.arange(num_images, dtype=np.int64)}

    splits = np.load(split_indices_path)
    train_indices = np.asarray(splits["train_indices"], dtype=np.int64)
    holdout_indices = np.asarray(splits["holdout_indices"], dtype=np.int64)
    if len(holdout_indices) == 0:
        return {
            "train": train_indices,
            "calib": np.array([], dtype=np.int64),
            "test": np.array([], dtype=np.int64),
        }

    num_calib = int(round(calib_ratio * len(holdout_indices)))
    return {
        "train": train_indices,
        "calib": holdout_indices[:num_calib],
        "test": holdout_indices[num_calib:],
    }


def _layer_dict(module: Any, index: int) -> dict[str, Any] | None:
    import torch

    if isinstance(module, torch.nn.Linear):
        return {
            "type": "linear",
            "index": index,
            "weight": module.weight.detach().cpu().numpy(),
            "bias": module.bias.detach().cpu().numpy(),
        }
    if isinstance(module, torch.nn.BatchNorm1d):
        return {
            "type": "batchnorm1d",
            "index": index,
            "weight": module.weight.detach().cpu().numpy(),
            "bias": module.bias.detach().cpu().numpy(),
            "running_mean": module.running_mean.detach().cpu().numpy(),
            "running_var": module.running_var.detach().cpu().numpy(),
            "eps": float(module.eps),
        }
    if isinstance(module, torch.nn.GELU):
        return {"type": "gelu", "index": index, "approximate": "none"}
    if isinstance(module, torch.nn.Dropout):
        return {"type": "dropout", "index": index, "p": float(module.p), "eval_identity": True}
    return None


def _export_model_layers(trainer: Any, out_dir: Path) -> list[dict[str, Any]]:
    model_net = trainer.net.module.model.model_net
    layers = []
    for idx, module in enumerate(model_net):
        layer = _layer_dict(module, idx)
        if layer is None:
            raise TypeError(f"Unsupported layer at model_net[{idx}]: {module!r}")
        layers.append(layer)
    np.save(out_dir / "model_layers.npy", np.array(layers, dtype=object), allow_pickle=True)
    return layers


def _export_scalers(trainer: Any, out_dir: Path) -> dict[str, Any]:
    target_mean = float(trainer.target_scaler.target_mean.detach().cpu().numpy())
    target_std = float(trainer.target_scaler.target_std.detach().cpu().numpy())
    np.savez(out_dir / "target_scaler.npz", target_mean=target_mean, target_std=target_std)

    feature_scaler = trainer.feature_scaler
    feature_meta: dict[str, Any] = {
        "transform": feature_scaler.transform,
        "elementwise": bool(feature_scaler.elementwise),
        "threshold": float(feature_scaler.threshold),
    }
    feature_arrays: dict[str, np.ndarray] = {}
    if feature_scaler.elementwise:
        feature_meta["unique_atomic_numbers"] = [int(value) for value in feature_scaler.unique]
        for atomic_number, scale in feature_scaler.scales.items():
            prefix = f"z{int(atomic_number)}"
            feature_arrays[f"{prefix}_offset"] = scale["offset"].detach().cpu().numpy()
            feature_arrays[f"{prefix}_scale"] = scale["scale"].detach().cpu().numpy()
    else:
        feature_arrays["offset"] = feature_scaler.scale["offset"].detach().cpu().numpy()
        feature_arrays["scale"] = feature_scaler.scale["scale"].detach().cpu().numpy()
    np.savez(out_dir / "feature_scaler.npz", **feature_arrays)
    _write_json(out_dir / "feature_scaler.json", feature_meta)

    return {"target_mean": target_mean, "target_std": target_std, "feature": feature_meta}


def _load_trainer_readonly(checkpoint_dir: Path) -> Any:
    """Load an AmpTorch trainer on CPU without writing ``params_cpu.pt``."""
    import torch
    from amptorch.trainer import AtomsTrainer

    checkpoint_dir = checkpoint_dir.resolve()
    cwd = Path.cwd()
    try:
        config = torch.load(checkpoint_dir / "config.pt", map_location="cpu")
        config["cmd"]["debug"] = True
        config["optim"]["gpus"] = 0

        trainer = AtomsTrainer(config)
        trainer.pretrained = True
        trainer.elements = config["dataset"]["descriptor"][-1]
        trainer.input_dim = config["dataset"]["fp_length"]
        trainer.load(load_dataset=False)
        trainer.net.initialize()

        state_dict = torch.load(checkpoint_dir / "params.pt", map_location=torch.device("cpu"))
        if state_dict and next(iter(state_dict)).startswith("module."):
            state_dict = OrderedDict((key[len("module.") :], value) for key, value in state_dict.items())
        trainer.net.module.load_state_dict(state_dict)

        normalizers = torch.load(checkpoint_dir / "normalizers.pt", map_location=torch.device("cpu"))
        trainer.feature_scaler = normalizers["feature"]
        trainer.target_scaler = normalizers["target"]
        trainer.net.module.eval()
        return trainer
    finally:
        os.chdir(cwd)


def _cached_data_from_h5(image: Any, descriptor_cache: Path, elements: list[str]) -> Any:
    import h5py
    import torch
    from torch_geometric.data import Data
    from amptorch.descriptor.util import get_hash

    image_hash = get_hash(image)
    h5_path = descriptor_cache / f"{image_hash}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Descriptor cache file not found: {h5_path}")

    symbols = np.array(image.get_chemical_symbols())
    num_atoms = len(symbols)
    atomic_numbers = torch.LongTensor(image.get_atomic_numbers())

    fp_by_element: dict[str, np.ndarray] = {}
    index_by_element: dict[str, np.ndarray] = {}
    num_desc_by_element: dict[str, int] = {}
    with h5py.File(h5_path, "r") as handle:
        group = handle["0"]
        for element in elements:
            if element not in symbols:
                continue
            if element not in group:
                raise KeyError(f"{h5_path} is missing element group {element!r}")
            element_group = group[element]
            fps = np.asarray(element_group["fps"])
            size_info = np.asarray(element_group["size_info"])
            fp_by_element[element] = fps
            index_by_element[element] = np.arange(num_atoms)[symbols == element]
            num_desc_by_element[element] = int(size_info[2])

    if not fp_by_element:
        raise ValueError(f"No cached fingerprints found for {h5_path}")

    num_desc_max = max(num_desc_by_element.values())
    image_fp_array = np.zeros((num_atoms, num_desc_max), dtype=np.float64)
    for element, fps in fp_by_element.items():
        num_desc = num_desc_by_element[element]
        image_fp_array[index_by_element[element], :num_desc] = fps

    return Data(
        fingerprint=torch.tensor(image_fp_array, dtype=torch.get_default_dtype()),
        atomic_numbers=atomic_numbers,
        num_nodes=num_atoms,
    )


def _convert_images_to_data(
    trainer: Any,
    images: list,
    descriptor_cache: Path | None,
    descriptor_cores: int,
) -> list:
    if descriptor_cache is not None:
        elements = [str(element) for element in trainer.config["dataset"]["descriptor"][-1]]
        data_list = [_cached_data_from_h5(image, descriptor_cache, elements) for image in images]
    else:
        from amptorch.dataset import construct_descriptor
        from amptorch.preprocessing import AtomsToData

        descriptor = construct_descriptor(trainer.config["dataset"]["descriptor"])
        a2d = AtomsToData(
            descriptor=descriptor,
            r_energy=False,
            r_forces=False,
            save_fps=False,
            fprimes=False,
            cores=descriptor_cores,
        )
        data_list = a2d.convert_all(images, disable_tqdm=True)

    trainer.feature_scaler.norm(data_list, disable_tqdm=True)
    return data_list


def _bias_corrections(images: list, element_bias: dict[str, Any]) -> np.ndarray:
    coeffs = element_bias["coefficients_ha"]
    corrections = []
    for image in images:
        correction = 0.0
        for symbol in image.get_chemical_symbols():
            if symbol not in coeffs:
                raise KeyError(f"No element-bias coefficient for symbol {symbol!r}")
            correction += float(coeffs[symbol])
        corrections.append(correction)
    return np.asarray(corrections, dtype=np.float64)


def _torch_latent(model_net: Any, fingerprints: Any, latent_layer: int) -> Any:
    layer_index = latent_layer if latent_layer >= 0 else len(model_net) + latent_layer
    x = fingerprints
    for idx, module in enumerate(model_net):
        x = module(x)
        if idx == layer_index:
            return x
    raise IndexError(f"latent layer {latent_layer} is out of range for {len(model_net)} layers")


def _segment_mean_numpy(values: np.ndarray, image_ids: np.ndarray, num_images: int) -> np.ndarray:
    sums = np.zeros((num_images, values.shape[1]), dtype=values.dtype)
    counts = np.zeros(num_images, dtype=values.dtype)
    np.add.at(sums, image_ids, values)
    np.add.at(counts, image_ids, 1)
    return sums / counts[:, None]


def _predict_dataset(
    trainer: Any,
    data_list: list,
    images: list,
    element_bias: dict[str, Any],
    batch_size: int,
    latent_layer: int,
) -> dict[str, np.ndarray]:
    import torch
    from amptorch.dataset import DataCollater

    collate_fn = DataCollater(train=False, forcetraining=False)
    model_net = trainer.net.module.model.model_net
    trainer.net.module.eval()

    atom_fps_parts = []
    image_id_parts = []
    atom_counts = []
    for image_id, data in enumerate(data_list):
        fps = data.fingerprint.detach().cpu().numpy()
        atom_fps_parts.append(fps)
        image_id_parts.append(np.full(fps.shape[0], image_id, dtype=np.int64))
        atom_counts.append(fps.shape[0])

    energy_norm_parts = []
    latent_parts = []
    for start in range(0, len(data_list), batch_size):
        batch_data = data_list[start : start + batch_size]
        batch = collate_fn(batch_data).to("cpu")
        energy_norm, _ = trainer.net.module([batch])
        latent_atoms = _torch_latent(model_net, batch.fingerprint, latent_layer)

        batch_image_ids = batch.batch.detach().cpu().numpy()
        latent_mean = _segment_mean_numpy(
            latent_atoms.detach().cpu().numpy(),
            batch_image_ids,
            num_images=len(batch_data),
        )
        energy_norm_parts.append(energy_norm.detach().cpu().numpy())
        latent_parts.append(latent_mean)

    energy_pred_normalized = np.concatenate(energy_norm_parts).astype(np.float64, copy=False)
    energy_pred_corrected = (
        trainer.target_scaler.denorm(torch.from_numpy(energy_pred_normalized), pred="energy")
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
    )
    bias = _bias_corrections(images, element_bias)
    energy_true = np.asarray([image.get_potential_energy() for image in images], dtype=np.float64)

    return {
        "atom_fps": np.concatenate(atom_fps_parts, axis=0).astype(np.float64, copy=False),
        "image_ids": np.concatenate(image_id_parts).astype(np.int64, copy=False),
        "atom_counts": np.asarray(atom_counts, dtype=np.int64),
        "energy_true_ha": energy_true,
        "energy_bias_correction_ha": bias,
        "energy_true_corrected_ha": energy_true - bias,
        "energy_pred_normalized": energy_pred_normalized,
        "energy_pred_corrected_ha": energy_pred_corrected,
        "energy_pred_ha": energy_pred_corrected + bias,
        "latent": np.concatenate(latent_parts, axis=0).astype(np.float64, copy=False),
    }


def _write_dataset(out_dir: Path, name: str, payload: dict[str, np.ndarray], source_indices: np.ndarray, images: list) -> None:
    np.save(out_dir / f"{name}_atom_fps.npy", payload["atom_fps"])
    np.save(out_dir / f"{name}_image_ids.npy", payload["image_ids"])
    np.save(out_dir / f"{name}_atom_counts.npy", payload["atom_counts"])
    np.save(out_dir / f"{name}_energy_true_ha.npy", payload["energy_true_ha"])
    np.save(out_dir / f"{name}_energy_bias_correction_ha.npy", payload["energy_bias_correction_ha"])
    np.save(out_dir / f"{name}_energy_true_corrected_ha.npy", payload["energy_true_corrected_ha"])
    np.save(out_dir / f"{name}_energy_pred_normalized.npy", payload["energy_pred_normalized"])
    np.save(out_dir / f"{name}_energy_pred_corrected_ha.npy", payload["energy_pred_corrected_ha"])
    np.save(out_dir / f"{name}_energy_pred_ha.npy", payload["energy_pred_ha"])
    np.save(out_dir / f"{name}_latent.npy", payload["latent"])
    np.save(out_dir / f"{name}_source_indices.npy", source_indices.astype(np.int64, copy=False))
    source_members = [str(image.info.get("source_member", "")) for image in images]
    np.save(out_dir / f"{name}_source_members.npy", np.asarray(source_members, dtype=object), allow_pickle=True)


def export(
    checkpoint_dir: Path,
    ase_path: Path,
    split_indices_path: Path | None,
    element_bias_path: Path,
    out_dir: Path,
    dataset_name: str = "all",
    calib_ratio: float = 0.5,
    max_images: int | None = None,
    sample_seed: int | None = None,
    descriptor_cache: Path | None = None,
    descriptor_cores: int = 1,
    batch_size: int = 256,
    latent_layer: int = DEFAULT_LATENT_LAYER,
    overwrite: bool = False,
) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_dir} is not empty; pass --overwrite to update it.")
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = checkpoint_dir.resolve()
    ase_path = ase_path.resolve()
    split_indices_path = split_indices_path.resolve() if split_indices_path else None
    element_bias_path = element_bias_path.resolve()
    descriptor_cache = descriptor_cache.resolve() if descriptor_cache else None

    trainer = _load_trainer_readonly(checkpoint_dir)
    layers = _export_model_layers(trainer, out_dir)
    scaler_metadata = _export_scalers(trainer, out_dir)
    element_bias = _load_json(element_bias_path)
    _write_json(out_dir / "element_bias.json", element_bias)

    images_all = _load_ase_images(ase_path)
    split_map = _split_indices(len(images_all), split_indices_path, dataset_name, calib_ratio)
    metadata: dict[str, Any] = {
        "checkpoint_dir": str(checkpoint_dir),
        "ase_path": str(ase_path),
        "split_indices_path": str(split_indices_path) if split_indices_path else None,
        "element_bias_path": str(element_bias_path),
        "descriptor_cache": str(descriptor_cache) if descriptor_cache else None,
        "latent_layer": latent_layer,
        "num_layers": len(layers),
        "target_scaler": scaler_metadata,
        "datasets": {},
    }

    for split_name, indices in split_map.items():
        indices = _select_indices(indices, max_images=max_images, sample_seed=sample_seed)
        if len(indices) == 0:
            metadata["datasets"][split_name] = {"num_images": 0, "skipped": True}
            continue

        images = [images_all[int(idx)] for idx in indices]
        data_list = _convert_images_to_data(
            trainer,
            images,
            descriptor_cache=descriptor_cache,
            descriptor_cores=descriptor_cores,
        )
        payload = _predict_dataset(
            trainer,
            data_list,
            images,
            element_bias=element_bias,
            batch_size=batch_size,
            latent_layer=latent_layer,
        )
        _write_dataset(out_dir, split_name, payload, indices, images)
        metadata["datasets"][split_name] = {
            "num_images": int(len(images)),
            "num_atoms": int(payload["atom_fps"].shape[0]),
            "fp_dim": int(payload["atom_fps"].shape[1]),
            "latent_dim": int(payload["latent"].shape[1]),
            "source_indices_path": f"{split_name}_source_indices.npy",
        }

    _write_json(out_dir / "export_metadata.json", metadata)
    return metadata


def _import_jax():
    from jax import config as jax_config

    jax_config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as jnp

    return jax, jnp


def _load_layers_for_jax(export_dir: Path) -> list[dict[str, Any]]:
    _, jnp = _import_jax()
    raw_layers = np.load(export_dir / "model_layers.npy", allow_pickle=True).tolist()
    params = []
    for layer in raw_layers:
        layer = dict(layer)
        for key in ("weight", "bias", "running_mean", "running_var"):
            if key in layer:
                layer[key] = jnp.asarray(layer[key], dtype=jnp.float64)
        params.append(layer)
    return params


def load_jax_model(export_dir: str | Path):
    """Load exported model arrays and return JAX-ready params and metadata."""
    export_dir = Path(export_dir)
    params = _load_layers_for_jax(export_dir)
    target_npz = np.load(export_dir / "target_scaler.npz")
    target_scaler = {
        "target_mean": float(target_npz["target_mean"]),
        "target_std": float(target_npz["target_std"]),
    }
    element_bias = _load_json(export_dir / "element_bias.json")
    return params, target_scaler, element_bias, jax_energy, jax_latent_fn


def jax_mlp(params: list[dict[str, Any]], atom_fps: Any, stop_at: int | None = None):
    jax, jnp = _import_jax()
    if stop_at is not None and stop_at < 0:
        stop_at = len(params) + stop_at

    x = jnp.asarray(atom_fps, dtype=jnp.float64)
    for idx, layer in enumerate(params):
        layer_type = layer["type"]
        if layer_type == "linear":
            x = x @ layer["weight"].T + layer["bias"]
        elif layer_type == "batchnorm1d":
            x = (x - layer["running_mean"]) / jnp.sqrt(layer["running_var"] + layer["eps"])
            x = x * layer["weight"] + layer["bias"]
        elif layer_type == "gelu":
            x = jax.nn.gelu(x, approximate=False)
        elif layer_type == "dropout":
            pass
        else:
            raise TypeError(f"Unsupported exported layer type {layer_type!r}")

        if stop_at is not None and idx == stop_at:
            return x
    return x


def jax_energy(params: list[dict[str, Any]], atom_fps: Any, image_ids: Any, num_images: int):
    jax, jnp = _import_jax()
    atom_energy = jnp.squeeze(jax_mlp(params, atom_fps), axis=-1)
    return jax.ops.segment_sum(
        atom_energy,
        jnp.asarray(image_ids, dtype=jnp.int32),
        num_segments=int(num_images),
    )


def jax_latent_fn(
    params: list[dict[str, Any]],
    atom_fps: Any,
    image_ids: Any,
    num_images: int,
    latent_layer: int = DEFAULT_LATENT_LAYER,
):
    jax, jnp = _import_jax()
    atom_latent = jax_mlp(params, atom_fps, stop_at=latent_layer)
    image_ids = jnp.asarray(image_ids, dtype=jnp.int32)
    sums = jax.ops.segment_sum(atom_latent, image_ids, num_segments=int(num_images))
    counts = jax.ops.segment_sum(
        jnp.ones((atom_latent.shape[0],), dtype=atom_latent.dtype),
        image_ids,
        num_segments=int(num_images),
    )
    return sums / counts[:, None]


def jax_denormalize_energy(normalized_energy: Any, target_scaler: dict[str, float], bias_corrections: Any | None = None):
    _, jnp = _import_jax()
    corrected = normalized_energy * target_scaler["target_std"] + target_scaler["target_mean"]
    if bias_corrections is None:
        return corrected
    return corrected + jnp.asarray(bias_corrections, dtype=corrected.dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--ase-path", default=str(DEFAULT_ASE_PATH))
    parser.add_argument(
        "--split-indices",
        default=None,
        help=f"Saved train/holdout split NPZ. Use {DEFAULT_SPLIT_INDICES} for the C4h_1 paper run.",
    )
    parser.add_argument("--element-bias", default=str(DEFAULT_ELEMENT_BIAS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--dataset-name", default="all")
    parser.add_argument("--calib-ratio", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument(
        "--descriptor-cache",
        default=None,
        help="Read-only descriptor H5 cache directory. Omit to recompute without saving fingerprints.",
    )
    parser.add_argument("--descriptor-cores", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-layer", type=int, default=DEFAULT_LATENT_LAYER)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = export(
        checkpoint_dir=Path(args.checkpoint_dir),
        ase_path=Path(args.ase_path),
        split_indices_path=Path(args.split_indices) if args.split_indices else None,
        element_bias_path=Path(args.element_bias),
        out_dir=Path(args.out_dir),
        dataset_name=args.dataset_name,
        calib_ratio=args.calib_ratio,
        max_images=args.max_images,
        sample_seed=args.sample_seed,
        descriptor_cache=Path(args.descriptor_cache) if args.descriptor_cache else None,
        descriptor_cores=args.descriptor_cores,
        batch_size=args.batch_size,
        latent_layer=args.latent_layer,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata["datasets"], indent=2))
    print(f"Wrote JAX export to {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
