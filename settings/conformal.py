from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, List
import json

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from utils.augs import Aug, _random_ortho, _id, _permute_coord, make_mmlu_cached_random_permute_aug
from utils.dabstats import DABStats, DABConfig, run_CIs, make_cis_gen, make_dab_ci_func

from utils.utils import batched_fori_loop

"""
    Helper functions
"""

MMLU_SUBJECTS = (
    "college_computer_science",
    "formal_logic",
    "high_school_computer_science",
    "computer_security",
    "machine_learning",
    "clinical_knowledge",
    # "high_school_biology",
    # "anatomy",
    # "college_chemistry",
    # "college_medicine",
    # "professional_medicine",
    # "business_ethics",
    # "professional_accounting",
    # "public_relations",
    # "management",
    # "marketing",
)

MMLU_ID_SCORES_DIR = Path(__file__).resolve().parents[1] / "data/mmlu/local_scores_ALL"
MMLU_RANDOM_PERMUTE_SCORES_DIR = Path(__file__).resolve().parents[1] / "data/mmlu/local_scores_AUGMENTED"
MMLU_TRANSFORMS = ("id", "random_permute")

def make_CP_stat(n_train, n, d):
    def CP_stat(data, key):
        # fit beta to training data
        xs_train = jax.lax.dynamic_slice(data, [0,0], [n_train,d])
        ys_train = jax.lax.dynamic_slice(data, [0,d], [n_train,1])
        XX = jnp.einsum('ij,it->jt', xs_train / jnp.sqrt(n_train), xs_train / jnp.sqrt(n_train))
        XY = jnp.einsum('ij,it->jt', xs_train / jnp.sqrt(n_train), ys_train / jnp.sqrt(n_train))
        hat_beta = jnp.linalg.lstsq(XX, XY)[0]
        # compute loss on last data point
        x_test = jax.lax.dynamic_slice(data, [n,0], [1,d])
        y_test = jax.lax.dynamic_slice(data, [n,d], [1,1])

        yy_test = jnp.dot(y_test, y_test)
        xy_test = jnp.dot(y_test, x_test @ hat_beta)
        xx_test = jnp.dot(x_test @ hat_beta, x_test @ hat_beta)

        return jnp.squeeze(yy_test - 2 * xy_test + xx_test)
    return CP_stat


def make_mmlu_CP_stat(n, d=4):
    def CP_stat(data, key):
        probs = data[n, :d]
        label = data[n, d].astype(jnp.int32)
        return 1.0 - probs[label]
    return CP_stat


def _mmlu_transform_names(transform_names):
    if transform_names is None:
        return ["id"]

    names = list(dict.fromkeys(transform_names))
    unknown = [name for name in names if name not in MMLU_TRANSFORMS]
    if unknown:
        raise ValueError(f"MMLU conformal transforms must be in {MMLU_TRANSFORMS}, found {unknown}")
    return names


def _load_mmlu_subject_arrays(scores_dir: Path, subject: str, d: int):
    scores_path = scores_dir / f"{subject}_scores.npy"
    targets_path = scores_dir / f"{subject}_targets.npy"
    metadata_path = scores_dir / f"{subject}_metadata.json"

    scores = np.load(scores_path)
    targets = np.load(targets_path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    eval_rows = metadata.get("eval_rows", [])

    if scores.ndim != 3 or scores.shape[-1] != d:
        raise ValueError(f"Expected {scores_path} with shape (num_prompts, num_questions, {d}) but found {scores.shape}")
    if targets.ndim != 1 or targets.shape[0] != scores.shape[1]:
        raise ValueError(f"Expected {targets_path} with shape ({scores.shape[1]},) but found {targets.shape}")
    if len(eval_rows) != scores.shape[1]:
        raise ValueError(f"Expected {metadata_path} to describe {scores.shape[1]} eval rows but found {len(eval_rows)}")

    avg_scores = scores.mean(axis=0).astype(np.float32)
    labels = targets.astype(np.int32)
    if not np.isfinite(avg_scores).all():
        raise ValueError(f"Non-finite MMLU probabilities found in {scores_path}")
    if np.any((labels < 0) | (labels >= d)):
        raise ValueError(f"MMLU targets in {targets_path} must be integers in [0, {d})")

    return avg_scores, labels, eval_rows


def _load_mmlu_id_rows(subjects: List[str], scores_dir: Path, d: int):
    rows = {}
    order = []

    for subject in subjects:
        avg_scores, labels, eval_rows = _load_mmlu_subject_arrays(scores_dir, subject, d)
        for row_idx, row in enumerate(eval_rows):
            key = (subject, row["split"], int(row["row_id"]))
            if key in rows:
                raise ValueError(f"Duplicate MMLU id row metadata: {key}")
            if int(row.get("target_idx", labels[row_idx])) != int(labels[row_idx]):
                raise ValueError(f"Target mismatch in id metadata for {key}")
            rows[key] = np.concatenate([avg_scores[row_idx], [float(labels[row_idx])]]).astype(np.float32)
            order.append(key)

    return order, rows


def _load_mmlu_random_permute_groups(subjects: List[str], scores_dir: Path, d: int):
    grouped_rows = {}

    for subject in subjects:
        avg_scores, labels, eval_rows = _load_mmlu_subject_arrays(scores_dir, subject, d)
        for row_idx, row in enumerate(eval_rows):
            for field in ("source_split", "source_row_id", "option_permutation"):
                if field not in row:
                    raise ValueError(f"Augmented metadata for {subject} is missing {field!r}")

            key = (subject, row["source_split"], int(row["source_row_id"]))
            permutation = str(row["option_permutation"])
            group = grouped_rows.setdefault(key, {})
            if permutation in group:
                raise ValueError(f"Duplicate option_permutation {permutation!r} for {key}")
            if int(row.get("target_idx", labels[row_idx])) != int(labels[row_idx]):
                raise ValueError(f"Target mismatch in augmented metadata for {key}, {permutation}")

            permutation_index = int(row.get("permutation_index", len(group)))
            group[permutation] = (
                permutation_index,
                np.concatenate([avg_scores[row_idx], [float(labels[row_idx])]]).astype(np.float32),
            )

    group_keys = list(grouped_rows)
    if not group_keys:
        raise ValueError(f"No augmented MMLU rows found in {scores_dir}")

    counts = np.array([len(grouped_rows[key]) for key in group_keys], dtype=np.int32)
    max_count = int(counts.max())
    group_data = np.zeros((len(group_keys), max_count, d + 1), dtype=np.float32)

    for group_idx, key in enumerate(group_keys):
        entries = sorted(grouped_rows[key].items(), key=lambda item: (item[1][0], item[0]))
        for perm_idx, (_permutation, (_permutation_index, row_data)) in enumerate(entries):
            group_data[group_idx, perm_idx] = row_data

    group_id_by_key = {key: group_idx for group_idx, key in enumerate(group_keys)}
    return group_id_by_key, group_data, counts


"""
    Main functions
"""

# Gaussian and Rademacher and CentredGamma, linear regression, l2 loss
def simulate_linear_regression(cfg: DABConfig, setting: str, calib_ratio: float=0.4) -> Dict[str, Dict[str, jnp.ndarray]]:
    """simulations"""
    # specify data 
    d = cfg.d
    
    # n has to be a static value since it defines the conformal prediction augmentation
    if len(cfg.n) > 1:
        raise ValueError(f'For conformal prediction, only one value of n can be supplied but {len(cfg.n)} found')
    n = cfg.n[0]

    if setting == 'Gaussian':
        def data_gen(_n, key, _unused):
            xkey, bkey, ykey = jax.random.split(key, 3)
            xs = jax.random.normal(xkey, shape=(n+1,d))  # +1 for an additional training data point
            beta = jax.random.normal(bkey, shape=(d,1))
            ys = xs @ beta + jax.random.normal(ykey, shape=(n+1,1))
            return jnp.concat([xs, ys], axis=1)
    elif setting == 'Rademacher':
        def data_gen(_n, key, _unused):
            xkey, bkey, ykey = jax.random.split(key, 3)
            xs = jax.random.rademacher(xkey, shape=(n+1,d))   # +1 for an additional training data point
            beta = jax.random.normal(bkey, shape=(d,1))
            ys = xs @ beta + jax.random.normal(ykey, shape=(n+1,1))
            return jnp.concat([xs, ys], axis=1)
    elif setting == 'CentredGamma':
        def data_gen(_n, key, _unused):
            xkey, bkey, ykey = jax.random.split(key, 3)
            xs = jax.random.gamma(xkey, a=1, shape=(n+1,d)) - 1  # +1 for an additional training data point
            beta = jax.random.normal(bkey, shape=(d,1))
            ys = xs @ beta + jax.random.normal(ykey, shape=(n+1,1))
            return jnp.concat([xs, ys], axis=1)
    else:
        raise ValueError(f'setting can only be one of "Gaussian", "Rademacher" or "CentredGamma", but {setting} found')
    
    # specify stats
    n_train = int(np.floor(n * (1 - calib_ratio)))

    CP_stat = make_CP_stat(n_train, n, d)

    # read transforms
    cp_transform_names = list(set(cfg.compose_transform_names))
    _cp_transforms = []
    for tn in cp_transform_names:
        if tn == 'id':
            _cp_transforms.append(_id)
        elif tn == 'random_ortho':
            _cp_transforms.append(_random_ortho)
        elif tn == 'permute_coord':
            _cp_transforms.append(_permute_coord)
        else:
            raise ValueError(f"{tn} is not a valid transformation name")

    # specify cycling 
    n_calib = n - n_train 

    cycle_fn_list = [
        lambda data, _key, shift=shift: jnp.concat([
            jax.lax.dynamic_slice(data, [0,0], [n_train,d+1]),  # training data unchanged
            jnp.roll(
                jax.lax.dynamic_slice(data, [n_train,0], [n_calib + 1,d+1]), 
                shift, 
                axis=0
            )    # calibration data and the last data point are rolled
        ], axis=0) 
        for shift in range(1, n_calib+1) 
    ]
    conformal_cycle_list = [Aug(dataset_lvl_aug=cycle_fn) for cycle_fn in cycle_fn_list]
    
    # conformal prediction composed with transforms
    cis_name_list = [f'CP_compose_{tn}' for tn in cp_transform_names]
    cp_transforms = [Aug(obs_lvl_aug=_cp_transform) for _cp_transform in _cp_transforms]
    cp_dabstats_over_transforms = [ 
        DABStats(
            stat = CP_stat,
            aug = [cycle.compose(cp_transform) for cycle in conformal_cycle_list],
            across_dataset_mode = 'enum'
        )   
        for cp_transform in cp_transforms
    ]

    cis_gen_list = [
        make_cis_gen(make_dab_ci_func(dabstats, cfg.quantile_method, cfg.one_sided), CP_stat, post_process=None, batch_size=cfg.batch_size)
        for dabstats in cp_dabstats_over_transforms
    ]

    assert len(cis_name_list) == len(cis_gen_list)

    return run_CIs(cfg, data_gen, cis_gen_list, cis_name_list)

# AMPTORCH
"""
    AmpTorch conformal prediction
"""

TRAIN_DATA_PREFIX = 'data/qm_sym_c4h_1_jax/c4h_1'
CALIB_DATA_PREFIX = 'data/qm_sym_c4h_2_jax/c4h_2'


def make_amptorch_CP_stat(
    n_train: int = None,
    k_neighbors: int = 10,
    eps: float = 1e-12,
    random_unsort: bool = False
):
    """
    Non-conformity score for AmpTorch conformal prediction.
    """

    latent_train = np.load(f"{TRAIN_DATA_PREFIX}_latent.npy")
    if n_train is not None:
        latent_train = latent_train[:n_train]
    
    n_train = latent_train.shape[0]
    latent_train_norms = jnp.linalg.norm(latent_train, axis=1)**2

    def CP_stat(data, key):
        # compute only on the first data (use conformal to then cycle things around)
        true_energy_calib = data[0, 0]
        pred_energy_calib = data[0, 1]
        latent_calib = data[0, 2:]

        residual = jnp.abs(pred_energy_calib - true_energy_calib)
        
        if random_unsort is False:
            knn_dist = jnp.mean(jnp.sort(
                            jnp.sqrt( jnp.maximum(latent_train_norms - 2 * jnp.dot(latent_train, latent_calib) + (jnp.linalg.norm(latent_calib)**2), eps) )
                        )[:k_neighbors])
        else:
            train_idxes = jax.random.choice(key, latent_train.shape[0], shape=(k_neighbors,), replace=False)
            latent_train_norms_subset = jnp.take(latent_train_norms, train_idxes, axis=0)
            latent_train_subset = jnp.take(latent_train, train_idxes, axis=0)
            knn_dist = jnp.mean(
                            jnp.sqrt( jnp.maximum(latent_train_norms_subset - 2 * jnp.dot(latent_train_subset, latent_calib) + (jnp.linalg.norm(latent_calib)**2), eps) )
                        )

        return residual / jnp.maximum(knn_dist, eps)

    return CP_stat


def simulate_amptorch_conformal(
    cfg: DABConfig,
    n_train: int = None,
    k_neighbors: int = 10,
    random_unsort: bool = False,
) -> Dict[str, Dict[str, jnp.ndarray]]:
    """Run DAB conformal simulations on exported QM-sym AmpTorch predictions."""
    if len(cfg.n) > 1:
        raise ValueError(f'For conformal prediction, only one value of n can be supplied but {len(cfg.n)} found')
    n_calib = cfg.n[0]
    assert cfg.d == 64

    # data generation
    energy_true_calib = jnp.asarray(np.load(f"{CALIB_DATA_PREFIX}_energy_true_ha.npy"))
    energy_pred_calib = jnp.asarray(np.load(f"{CALIB_DATA_PREFIX}_energy_pred_ha.npy"))
    latent_calib = jnp.asarray(np.load(f"{CALIB_DATA_PREFIX}_latent.npy"))
    if energy_true_calib.ndim == 1:
        energy_true_calib = energy_true_calib[:, None]
    if energy_pred_calib.ndim == 1:
        energy_pred_calib = energy_pred_calib[:, None]

    num_available = int(energy_true_calib.shape[0])

    n_rows = n_calib + 1
    max_start = num_available - n_calib

    energy_true_dim = energy_true_calib.shape[1]
    energy_pred_dim = energy_pred_calib.shape[1]
    latent_dim = latent_calib.shape[1]

    def data_gen(n, key, _idx_unused):
        del n
        start_idx = jax.random.randint(key, (), 0, max_start)
        return jnp.concatenate(
            [
                jax.lax.dynamic_slice(energy_true_calib, (start_idx, 0), (n_rows, energy_true_dim)),
                jax.lax.dynamic_slice(energy_pred_calib, (start_idx, 0), (n_rows, energy_pred_dim)),
                jax.lax.dynamic_slice(latent_calib, (start_idx, 0), (n_rows, latent_dim)),
            ],
            axis=1,
        )
    
    # CP stat
    CP_stat = make_amptorch_CP_stat(
        n_train=n_train,
        k_neighbors=k_neighbors,
        random_unsort=random_unsort,
    )

    # conformal transforms
    cycle_fn_list = [
        lambda data, _key, shift=shift: jnp.roll(data, shift, axis=0)
        for shift in range(1, n_calib + 1)
    ]
    conformal_cycle_list = [Aug(dataset_lvl_aug=cycle_fn) for cycle_fn in cycle_fn_list]

    # read transforms
    cp_transform_names = list(set(cfg.compose_transform_names))
    _cp_transforms = []
    for tn in cp_transform_names:
        if tn == 'id':
            _cp_transforms.append(_id)
        else:
            raise ValueError(
                "simulate_amptorch_conformal currently supports only compose_transform_names=['id']; "
                f"{tn!r} would transform energy and latent columns without transforming the fixed "
                "training latent reference."
            )

    cis_name_list = [f"CP_compose_{tn}" for tn in cp_transform_names]
    cp_transforms = [Aug(obs_lvl_aug=_cp_transform) for _cp_transform in _cp_transforms]
    cp_dabstats_over_transforms = [
        DABStats(
            stat=CP_stat,
            aug=[cycle.compose(cp_transform) for cycle in conformal_cycle_list],
            across_dataset_mode="enum",
        )
        for cp_transform in cp_transforms
    ]

    cis_gen_list = [
        make_cis_gen(
            make_dab_ci_func(dabstats, cfg.quantile_method, cfg.one_sided),
            CP_stat,
            post_process=None,
            batch_size=cfg.batch_size,
        )
        for dabstats in cp_dabstats_over_transforms
    ]

    assert len(cis_name_list) == len(cis_gen_list)
    return run_CIs(cfg, data_gen, cis_gen_list, cis_name_list)

# Text


def simulate_mmlu_conformal(
    cfg: DABConfig,
    subjects: List[str],
    scores_dir: str = None,
) -> str:
    """Run DAB conformal prediction on cached MMLU option probabilities.

    Rows are ``(p_A, p_B, p_C, p_D, label[, augmented_group_id])``.
    """
    if subjects is None:
        raise ValueError("At least one MMLU subject must be supplied")

    if len(cfg.n) != 1:
        raise ValueError(f'For conformal prediction, only one value of n can be supplied but {len(cfg.n)} found')
    if cfg.d != 4:
        raise ValueError(f'MMLU conformal prediction requires cfg.d == 4 but {cfg.d} found')
    for subject in subjects:
        if subject not in MMLU_SUBJECTS:
            raise ValueError(f"Unknown MMLU subject {subject!r}")

    d = cfg.d
    n = cfg.n[0]
    transform_names = _mmlu_transform_names(cfg.compose_transform_names)
    needs_random_permute = "random_permute" in transform_names

    id_scores_dir = Path(scores_dir) if scores_dir is not None else MMLU_ID_SCORES_DIR
    id_order, id_rows = _load_mmlu_id_rows(subjects, id_scores_dir, d)

    aug_group_data = None
    aug_counts = None
    if needs_random_permute:
        group_id_by_key, aug_group_data_np, aug_counts_np = _load_mmlu_random_permute_groups(
            subjects,
            MMLU_RANDOM_PERMUTE_SCORES_DIR,
            d,
        )
        missing_keys = [key for key in id_order if key not in group_id_by_key]
        if missing_keys:
            raise ValueError(f"Augmented MMLU cache is missing {len(missing_keys)} source rows; first missing key: {missing_keys[0]}")

        data_rows = [
            np.concatenate([id_rows[key], [float(group_id_by_key[key])]]).astype(np.float32)
            for key in id_order
        ]
        aug_group_data = jnp.asarray(aug_group_data_np, dtype=jnp.float32)
        aug_counts = jnp.asarray(aug_counts_np, dtype=jnp.int32)
    else:
        data_rows = [
            np.concatenate([id_rows[key], [-1.0]]).astype(np.float32)
            for key in id_order
        ]

    all_data = jnp.asarray(np.stack(data_rows), dtype=jnp.float32)
    N = all_data.shape[0]
    if N < n + 1:
        raise ValueError(f"MMLU cache has only {N} usable scored rows; cfg.n[0] + 1 requires {n + 1}")

    def data_gen(_n_unused, key, _idx_unused):
        idx = jax.random.choice(key, N, shape=(n + 1,), replace=False)
        return all_data[idx]

    CP_stat = make_mmlu_CP_stat(n, d)

    # specify cycling 
    # - augmentations are done before local generation and caching of scores. No augmentations are done on the fly
    cycle_fn_list = [
        lambda data, _key, shift=shift: jnp.roll(
                data,
                shift,
                axis=0
            ) 
        for shift in range(1, n+1)
    ]
    conformal_cycle_list = [Aug(dataset_lvl_aug=cycle_fn) for cycle_fn in cycle_fn_list]
    cis_name_list = []
    cis_gen_list = []
    random_permute_aug = (
        Aug(dataset_lvl_aug=make_mmlu_cached_random_permute_aug(aug_group_data, aug_counts, d))
        if needs_random_permute
        else None
    )
    for transform_name in transform_names:
        cis_name_list.append(transform_name)
        if transform_name == "id":
            dabstats = DABStats(
                stat = CP_stat,
                aug = [cycle for cycle in conformal_cycle_list],
                across_dataset_mode = 'enum'
            )
            ci_func = make_dab_ci_func(dabstats, cfg.quantile_method, cfg.one_sided)
        elif transform_name == "random_permute":
            dabstats = DABStats(
                stat = CP_stat,
                aug = [cycle.compose(random_permute_aug) for cycle in conformal_cycle_list],
                across_dataset_mode = 'enum'
            )
            ci_func = make_dab_ci_func(dabstats, cfg.quantile_method, cfg.one_sided)
        else:
            raise ValueError(f"{transform_name} is not a valid MMLU transform name")
        cis_gen_list.append(
            make_cis_gen(ci_func, CP_stat, post_process=None, batch_size=cfg.batch_size)
        )

    return run_CIs(cfg, data_gen, cis_gen_list, cis_name_list)
