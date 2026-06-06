
"""Augmentation utilities and deterministic variants."""

import itertools
import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jndimage
import numpy as np

from functools import partial
from typing import Any, Self
from utils.utils import concat_xs_ys_ws, split_xs_ys_ws

def _enumerate_dataset_lvl(func_list):
    """Modify a per-dataset augmentation to produces (xs) enumerated over a list of deterministic augmentations
    """
    count = len(func_list)
    @partial(jax.jit, static_argnames=("num",))
    def vmapped(xs: jnp.ndarray, key: jax.Array, num: int) -> jnp.ndarray:
        idxs = jnp.arange(count * num, dtype=jnp.int32)
        subkeys = jax.random.split(key, count * num)
        return jax.vmap(
            lambda idx, subkey: jax.lax.switch(idx % count, func_list, xs, subkey),
            in_axes=(0, 0),
        )(idxs, subkeys)
    return vmapped

def _iid_dataset_lvl(func):
    """Lift a per-dataset augmentation to accept (xs, key, num).

    The function is vmapped over `num` subkeys from `key`, so callers
    use the dataset-level signature even if the inner function only consumes
    a single subkey.
    """
    @partial(jax.jit, static_argnames=("num",))
    def vmapped(xs: jnp.ndarray, key: jax.Array, num: int) -> jnp.ndarray:
        return jax.vmap(lambda subkey: func(xs, subkey))(jax.random.split(key, num))

    return vmapped

def _iid_obs_lvl(func):
    """Lift a per-observation augmentation to accept (xs, subkey).

    The function is vmapped over observations in `xs` using per-item
    keys derived from `subkey`.
    """
    def vmapped(xs: jnp.ndarray, subkey: jax.Array) -> jnp.ndarray:
        return jax.vmap(
            lambda x, x_key: func(x, x_key), in_axes=(0, 0)
        )(xs, jax.random.split(subkey, xs.shape[0]))

    return vmapped

def _double_obs_with_weight(func):
    """Lift a per-observation augmentation to accept (x_y_w, key), where x (data1) and y(data2) are separated augmented while w (weight) is unchanged
    """
    def func_w_weight(x_y_w: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        assert len(x_y_w.shape) == 1 and x_y_w.shape[0] % 2 == 1
        d = x_y_w.shape[0] // 2
        x = x_y_w[:d]
        y = x_y_w[d:2*d]
        w = x_y_w[-1:]

        x_key, y_key = jax.random.split(key)
        new_x = func(x, x_key)
        new_y = func(y, y_key)

        return jnp.concat([new_x, new_y, w])
    return func_w_weight


def _double_obs_single_aug_with_weight(func):
    """Lift a per-observation augmentation to accept (x_y_w, key), where x (data1) is augmented but y(data2) is not, and w (weight) is unchanged
    """
    def func_w_weight(x_y_w: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        assert len(x_y_w.shape) == 1 and x_y_w.shape[0] % 2 == 1
        d = x_y_w.shape[0] // 2
        x = x_y_w[:d]
        y = x_y_w[d:2*d]
        w = x_y_w[-1:]

        new_x = func(x, key)

        return jnp.concat([new_x, y, w])
    return func_w_weight

class Aug:
    """
    Augmentation class
    
    :var obs_lvl_aug: function (x: jnp.ndarray, key: jax.Array) -> jnp.ndarray
    :var dataset_lvl_aug: function (xs: jnp.ndarray, key: jax.Array) -> jnp.ndarray, where xs is of shape (n,...)
    """
    def __init__(self,
                 obs_lvl_aug = None,
                 dataset_lvl_aug = None,
                 ):
        if (obs_lvl_aug is None and dataset_lvl_aug is None) or (obs_lvl_aug is not None and dataset_lvl_aug is not None):
            raise ValueError("Exactly one of obs_lvl_aug and dataset_lvl_aug should be specified")
        self.obs_lvl_aug = obs_lvl_aug
        
        if obs_lvl_aug is None:
            self.dataset_lvl_aug = dataset_lvl_aug
        else: 
            self.dataset_lvl_aug = _iid_obs_lvl(obs_lvl_aug)
        
        
    def compose(self, aug2: Self):
        # compose augmentations at the observation level
        if (self.obs_lvl_aug is None) or (aug2.obs_lvl_aug is None):
            obs_lvl_aug = None 
            def dataset_lvl_aug(xs: jnp.ndarray, key: jax.Array):
                key0, key1 = jax.random.split(key, 2)
                return self.dataset_lvl_aug(aug2.dataset_lvl_aug(xs, key0), key1)
        else:
            def obs_lvl_aug(x: jnp.ndarray, key: jax.Array):
                key0, key1 = jax.random.split(key, 2)
                return self.obs_lvl_aug(aug2.obs_lvl_aug(x, key0), key1)
            dataset_lvl_aug = None

        return Aug(
            obs_lvl_aug=obs_lvl_aug,
            dataset_lvl_aug=dataset_lvl_aug
        )

"""
    Augmentations at observation or dataset level
"""

def _id(x: jnp.ndarray, x_key: jax.Array) -> jnp.ndarray:
    """identity
    """
    return x


def _bootstrap_uncentred(xs: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """Bootstrap resampling on a (n,d) dataset and then centering by empirical mean
    """
    indices = jax.random.choice(key, xs.shape[0], shape=(xs.shape[0],), replace=True)
    return jnp.take(xs, indices, axis=0) 


def _bootstrap(xs: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """Bootstrap resampling on a (n,d) dataset and then centering by empirical mean
    """
    return _bootstrap_uncentred(xs, key) - xs.mean(axis=0)

def _random_ortho(x: jnp.ndarray, x_key: jax.Array) -> jnp.ndarray:
    """Uniformly drawn orthogonal transformation per observation.
    """
    return jax.random.orthogonal(x_key, x.shape[0]) @ x

def _wild_bootstrap(xs_ys_ws: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """Wild bootstrap with rademacher weights on a (n,d) dataset pre-attached with (n, 1) weights
    """
    xs, ys, _ = split_xs_ys_ws(xs_ys_ws)
    ws = jax.random.rademacher(key, shape=(xs.shape[0], 1))
    return concat_xs_ys_ws(xs,ys, ws)

def _double_bootstrap(xs_ys_ws: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """Bootstrap resampling on two separate datasets where the input takes the same shape as _wild_bootstrap
    """
    xs, ys, ws = split_xs_ys_ws(xs_ys_ws)
    indices = jax.random.choice(key, xs.shape[0], shape=(xs.shape[0],), replace=True)
    bt_xs = jnp.take(xs, indices, axis=0)
    bt_ys = jnp.take(ys, indices, axis=0) # same indices must be used
    return concat_xs_ys_ws(bt_xs, bt_ys, ws)


def _permute_coord(x: jnp.ndarray, x_key: jax.Array) -> jnp.ndarray:
    """Randomly permute the coordinates of a data point
    """
    return jax.random.permutation(x_key, x, axis=0)


"""
    Augmentations for MMLU
"""

_MMLU_OPTION_LABELS = ("A", "B", "C", "D")


def _mmlu_label_to_letter(label: Any) -> str:
    if isinstance(label, str):
        normalized = label.strip().upper()
        if normalized in _MMLU_OPTION_LABELS:
            return normalized
    else:
        index = int(label)
        if 0 <= index < len(_MMLU_OPTION_LABELS):
            return _MMLU_OPTION_LABELS[index]
    raise ValueError(f"Expected an MMLU label in A/B/C/D or 0/1/2/3, got {label!r}")


def _mmlu_letter_to_label(letter: str, original_label: Any) -> str | int:
    if isinstance(original_label, str):
        return letter
    return _MMLU_OPTION_LABELS.index(letter)


def _mmlu_option_permute(x: tuple[dict[str, Any], Any]) -> list[tuple[dict[str, Any], str | int]]:
    """Enumerate all option permutations for one MMLU prompt/label pair.

    ``x`` is ``(prompt, label)`` where ``prompt`` has keys ``input`` and
    ``A``/``B``/``C``/``D``. The returned list has 24 ``(prompt, label)`` pairs
    and keeps the label pointing to the same answer text after relabeling.
    """

    prompt, label = x
    missing = [option for option in _MMLU_OPTION_LABELS if option not in prompt]
    if missing:
        raise ValueError(f"MMLU prompt is missing option keys: {missing}")

    original_label = _mmlu_label_to_letter(label)
    original_options = {option: prompt[option] for option in _MMLU_OPTION_LABELS}
    augmented = []

    for permutation in itertools.permutations(_MMLU_OPTION_LABELS):
        new_prompt = dict(prompt)
        for new_option, old_option in zip(_MMLU_OPTION_LABELS, permutation):
            new_prompt[new_option] = original_options[old_option]
        new_label_letter = _MMLU_OPTION_LABELS[permutation.index(original_label)]
        new_label = _mmlu_letter_to_label(new_label_letter, label)
        if "target" in new_prompt:
            new_prompt["target"] = new_label
        augmented.append((new_prompt, new_label))

    return augmented


def make_mmlu_cached_random_permute_aug(
    group_data: jnp.ndarray,
    counts: jnp.ndarray,
    d: int = 4,
):
    """Swap each MMLU row for one cached option-permutation variant.

    ``group_data`` has shape ``(num_source_rows, max_permutations, d + 1)`` and
    stores ``(p_A, p_B, p_C, p_D, label)`` rows. Input rows are expected to
    carry the source-group id at column ``d + 1``. Each call samples
    independently, so repeated DAB draws are with replacement.
    """

    def random_permute(data: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        group_ids = data[:, d + 1].astype(jnp.int32)
        row_keys = jax.random.split(key, data.shape[0])

        def sample_group(group_id, row_key):
            count = counts[group_id]
            perm_idx = jnp.floor(
                jax.random.uniform(row_key) * count
            ).astype(jnp.int32)
            row = group_data[group_id, perm_idx]
            return jnp.concat([row, group_id.astype(row.dtype)[None]])

        return jax.vmap(sample_group)(group_ids, row_keys)

    return random_permute


"""
    Augmentations for FermiNet
"""

def _rotate_elec(x: jnp.ndarray, x_key: jax.Array) -> jnp.ndarray:
    """Rotate all electron coordinates by one random angle in the xy plane.

    Args:
        x: Array of shape (n*3).
        x_key: PRNG key used to draw one uniform angle.
    """
    theta = jax.random.uniform(x_key, minval=0.0, maxval=2.0 * jnp.pi)
    c, s = jnp.cos(theta), jnp.sin(theta)
    rot = jnp.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return (x.reshape((-1,3)) @ rot.T).reshape((-1,))

"""
    Augmentations for MNIST
"""


def _sample_image_at_coordinates(image: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
    """Bilinear interpolation from floating-point coordinates."""
    sampled = jax.vmap(
        lambda image_per_channel: jndimage.map_coordinates(
            image_per_channel,
            [coords[0], coords[1]],
            order=1,
            mode="constant",
            cval=0.0,
        ), in_axes=-1
    )(image)
    return sampled.swapaxes(1,0).reshape(image.shape)


def _make_random_rotate(angle_min=-15, angle_max=15, num_channels=1):
    def _random_rotate(x: jnp.ndarray, x_key: jax.Array) -> jnp.ndarray:
        """Rotate x by a random angle in [-15,15] degrees after square reshape."""
        d = x.shape[0]
        w = round(np.sqrt(d / num_channels))
        assert len(x.shape) == 1 and d == w * w * num_channels
        image = x.reshape([w, w, num_channels])

        angle_deg = jax.random.uniform(x_key, minval=angle_min, maxval=angle_max)
        angle = jnp.deg2rad(angle_deg)
        cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
        rotation_inv = jnp.array([[cos_a, sin_a], [-sin_a, cos_a]])

        ys, xs = jnp.meshgrid(jnp.arange(w), jnp.arange(w), indexing="ij")
        coords = jnp.stack([ys.reshape(-1), xs.reshape(-1)], axis=0)
        center = jnp.array([(w - 1.0) / 2.0, (w - 1.0) / 2.0]).reshape(2, 1)

        src = rotation_inv @ (coords - center) + center
        rotated = _sample_image_at_coordinates(image, src)
        return rotated.reshape(-1)
    return _random_rotate



def _make_random_zoom(zoom_min=0.9, zoom_max=1.1, num_channels=1):
    def _random_zoom(x: jnp.ndarray, x_key: jax.Array) -> jnp.ndarray:
        """Zoom x in/out around center after square reshape."""
        d = x.shape[0]
        w = round(np.sqrt(d / num_channels))
        assert len(x.shape) == 1 and d == w * w * num_channels
        image = x.reshape([w, w, num_channels])

        scale = jax.random.uniform(x_key, minval=zoom_min, maxval=zoom_max)
        ys, xs = jnp.meshgrid(jnp.arange(w), jnp.arange(w), indexing="ij")
        coords = jnp.stack([ys.reshape(-1), xs.reshape(-1)], axis=0)
        center = jnp.array([(w - 1.0) / 2.0, (w - 1.0) / 2.0]).reshape(2, 1)

        # Inverse mapping from output grid to source image for zoom.
        src = (coords - center) / scale + center
        zoomed = _sample_image_at_coordinates(image, src)
        return zoomed.reshape(-1)
    return _random_zoom

def _make_random_rotate_zoom(angle_min=-15, angle_max=15, zoom_min=0.9, zoom_max=1.1, num_channels=1):
    _random_rotate = _make_random_rotate(angle_min, angle_max, num_channels=num_channels)
    _random_zoom = _make_random_zoom(zoom_min, zoom_max, num_channels=num_channels)
    def _random_rotate_zoom(x: jnp.ndarray, x_key: jax.Array) -> jnp.ndarray:
        rot_key, zoom_key = jax.random.split(x_key, 2)
        return _random_zoom(
                    _random_rotate(x, rot_key), 
                    zoom_key,
                )
    return _random_rotate_zoom


def _make_higgs_rotate(angle_min=-180, angle_max=180):

    HIGGS_FEATURE_NAMES = [
        # "class",
        "lepton pT", "lepton eta", "lepton phi", 
        "missing energy magnitude", "missing energy phi", 
        "jet 1 pt", "jet 1 eta", "jet 1 phi", "jet 1 b-tag", 
        "jet 2 pt", "jet 2 eta", "jet 2 phi", "jet 2 b-tag", 
        "jet 3 pt", "jet 3 eta", "jet 3 phi", "jet 3 b-tag", 
        "jet 4 pt", "jet 4 eta", "jet 4 phi", "jet 4 b-tag",
        "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"
    ]

    angle_features = [
        "lepton phi",
        "missing energy phi",
        "jet 1 phi",
        "jet 2 phi",
        "jet 3 phi",
        "jet 4 phi",
    ]
    angle_idx = jnp.asarray([
        HIGGS_FEATURE_NAMES.index(f) for f in angle_features
    ], dtype=jnp.int32)

    def wrapped_phi_shift(phi, angle):
        return (phi + angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

    def _random_rotate(x: jnp.ndarray, x_key: jax.Array) -> jnp.ndarray:
        """move all phi's by a random angle."""
        
        angle_deg = jax.random.uniform(x_key, minval=angle_min, maxval=angle_max)
        angle = jnp.deg2rad(angle_deg)

        shifted_angles = wrapped_phi_shift(x[angle_idx], angle)
        return x.at[angle_idx].set(shifted_angles)
    return _random_rotate
