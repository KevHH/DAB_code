import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import scipy
import functools


from skimage.transform import resize


from utils.augs import Aug, _wild_bootstrap, _random_ortho, _id, _double_obs_with_weight, _permute_coord, _make_random_zoom, _make_random_rotate, _make_random_rotate_zoom, _make_higgs_rotate
from utils.dabstats import DABStats, DABConfig, run_CIs, make_cis_gen, make_dab_ci_func
from utils.utils import concat_xs_ys_ws, split_xs_ys_ws, batched_fori_loop

"""
    Helper functions
"""

def make_rbf(bandwidth_squared):
    def rbf(x, y):
        x_sq = jnp.dot(x, x)
        y_sq = jnp.dot(y, y)
        x_y = jnp.dot(x, y)
        return jnp.exp( - (x_sq - 2 * x_y + y_sq ) / bandwidth_squared )
    return rbf

def make_mmd_kernel(kernel):
    return lambda x1, x2, y1, y2: kernel(x1,x2) + kernel(y1,y2) - kernel(x1,y2) - kernel(y1, x2)

def get_bandwidth(data):
    n = data.shape[0]
    assert n % 2 == 0
    consecutive_dist = jnp.linalg.norm(
            data[:(n // 2)] - data[(n // 2):], axis=-1
        )
    return jnp.median(consecutive_dist)

def get_weighted_data(xs_ys_ws):
    xs, ys, ws = split_xs_ys_ws(xs_ys_ws)
    weighted_xs = xs * ws
    weighted_ys = ys * ws
    return weighted_xs, weighted_ys

def compute_ustat(kernel, weighted_xs, weighted_ys, batch_size=None):
    n = weighted_xs.shape[0]

    @batched_fori_loop(batch_size=batch_size)
    def batched_vstat(batched_xs, batched_ys):
        return jax.vmap(
                    lambda x1, y1: jnp.sum(jax.vmap(
                        lambda x2, y2: kernel(x1, x2, y1, y2) / (n**2)
                    )(weighted_xs, weighted_ys))
            )(batched_xs, batched_ys)  # return one value per outer element

    vstat = jnp.sum(batched_vstat(weighted_xs, weighted_ys))
    diag = jnp.sum(jax.vmap(
        lambda x, y: kernel(x, x, y, y) / (n**2)
    )(weighted_xs, weighted_ys))
    ustat = (vstat - diag) * ( (n**2) / jnp.sqrt(n * (n-1)) )

    return ustat

# def compute_ustat(kernel, weighted_xs, weighted_ys, batch_size=None):
#     n = weighted_xs.shape[0]
#     del batch_size

#     kernel_matrix = jax.vmap(
#         lambda x1, y1: jax.vmap(
#             lambda x2, y2: kernel(x1, x2, y1, y2)
#         )(weighted_xs, weighted_ys)
#     )(weighted_xs, weighted_ys)

#     off_diag_sum = jnp.sum(kernel_matrix) - jnp.trace(kernel_matrix)
#     return off_diag_sum / jnp.sqrt(n * (n - 1))

def make_mmd(mmd_kernel, batch_size=None):
    def mmd(xs_ys_ws: jnp.ndarray, key: jax.Array):
        weighted_xs, weighted_ys = get_weighted_data(xs_ys_ws)
        ustat = compute_ustat(
            kernel=mmd_kernel,
            weighted_xs=weighted_xs,
            weighted_ys=weighted_ys,
            batch_size=batch_size
        )
        return ustat
    return mmd

def make_ave_mmd(mmd_kernel, aug, num_ave_transforms, batch_size=None):
    def mmd(xs_ys_ws: jnp.ndarray, key: jax.Array):
        weighted_xs, weighted_ys = get_weighted_data(xs_ys_ws)

        keys = jax.random.split(key, num_ave_transforms * 2)
        keys1 = keys[:num_ave_transforms]
        keys2 = keys[num_ave_transforms:]

        fixed_kernel = lambda x1, y1, x2, y2: jax.vmap(
                                                lambda key1: jax.vmap(
                                                    lambda key2: mmd_kernel(
                                                        aug.obs_lvl_aug(x1, key1), aug.obs_lvl_aug(x2, key2), y1, y2
                                                    )
                                                )(keys2).mean(axis=0)
                                            )(keys1).mean(axis=0)

        ustat = compute_ustat(
            kernel=fixed_kernel,
            weighted_xs=weighted_xs,
            weighted_ys=weighted_ys,
            batch_size=batch_size
        )
        return ustat
    return mmd

def WB_post_process(lower, upper, emp_est):
    """
        Wild bootstrap CI is for emp_est - truth, so we transform it into a CI for truth directly
    """
    return (lower - emp_est, upper - emp_est)

"""
    Main functions
"""

# Gaussian and Rademacher and CentredGamma, RBF kernel
def simulate_rbf_kernel(cfg: DABConfig, setting: str) -> Dict[str, Dict[str, jnp.ndarray]]:
    """simulations"""
    # specify data and stats
    d = cfg.d
    assert isinstance(cfg.testparam, list) and len(cfg.testparam) == 1
    shift_vec = cfg.testparam[0]
    scale_vec = cfg.testparam[0]
    # shift_vec = jnp.concat([jnp.array([cfg.testparam]), jnp.zeros(d-1) ], axis=0)
    # scale_vec = jnp.concat([jnp.array([cfg.testparam]), jnp.ones(d-1) ], axis=0)
    if setting == 'GaussianShift':
        def data_gen(n, key, _unused):
            data = jax.random.normal(key, shape=(n,2*d)) / jnp.sqrt(2)
            # normalisation chosen since Var[Z1-Z2]=2  for Z1, Z2 iid N(0,1)
            xs = data[:, :d]
            ys = data[:, d:(2*d)] + shift_vec
            return concat_xs_ys_ws(xs, ys, jnp.ones((n,1)))
    elif setting == 'GaussianScale':
        def data_gen(n, key, _unused):
            data = jax.random.normal(key, shape=(n,2*d)) / jnp.sqrt(2)
            # normalisation chosen since Var[Z1-Z2]=2  for Z1, Z2 iid N(0,1)
            xs = data[:, :d]
            ys = data[:, d:(2*d)] * scale_vec
            return concat_xs_ys_ws(xs, ys, jnp.ones((n,1)))
    elif setting == 'RademacherShift':
        def data_gen(n, key, _unused):
            data = jax.random.rademacher(key, shape=(n,2*d)) / jnp.sqrt(2)
            # normalisation chosen since Var[Y1-Y2]=2  for Y1, Y2 iid Rad
            xs = data[:, :d]
            ys = data[:, d:(2*d)] + shift_vec
            return concat_xs_ys_ws(xs, ys, jnp.ones((n,1)))
    elif setting == 'RademacherScale':
        def data_gen(n, key, _unused):
            data = jax.random.rademacher(key, shape=(n,2*d)) / jnp.sqrt(2)
            # normalisation chosen since Var[Y1-Y2]=2  for Y1, Y2 iid Rad
            xs = data[:, :d]
            ys = data[:, d:(2*d)] * scale_vec
            return concat_xs_ys_ws(xs, ys, jnp.ones((n,1)))
    elif setting == 'CentredGammaShift':
        def data_gen(n, key, _unused):
            data = (jax.random.gamma(key, a=1, shape=(n,2*d)) - 1) / jnp.sqrt(2)
            # normalisation chosen since Var[Y1-Y2]=2  for Y1, Y2 iid Rad
            xs = data[:, :d]
            ys = data[:, d:(2*d)] + shift_vec
            return concat_xs_ys_ws(xs, ys, jnp.ones((n,1)))
    elif setting == 'CentredGammaScale':
        def data_gen(n, key, _unused):
            data = (jax.random.gamma(key, a=1, shape=(n,2*d)) - 1) / jnp.sqrt(2)
            # normalisation chosen since Var[Y1-Y2]=2  for Y1, Y2 iid Rad
            xs = data[:, :d]
            ys = data[:, d:(2*d)] * scale_vec
            return concat_xs_ys_ws(xs, ys, jnp.ones((n,1)))
    else:
        raise ValueError(f'setting can only be one of "GaussianShift", "GaussianScale", "RademacherShift", "RademacherScale", "CentredGammaShift" or "CentredGammaScale", but {setting} found')
    
    rbf_kernel = make_mmd_kernel(
        kernel=make_rbf(d)
    )
    mmd_with_rbf_kernel = make_mmd(
        mmd_kernel=rbf_kernel,
    )

    # read transforms
    wb_transform_names = cfg.compose_transform_names
    _wb_transforms = []
    for tn in wb_transform_names:
        if tn == 'id':
            _wb_transforms.append(_id)
        elif tn == 'random_ortho':
            _wb_transforms.append(_double_obs_with_weight(_random_ortho))
        elif tn == 'permute_coord':
            _wb_transforms.append(_double_obs_with_weight(_permute_coord))
        else:
            raise ValueError(f"{tn} is not a valid transformation name")

    # wild bootstrap DAB-compose transforms
    cis_name_list = [f'WB_compose_{tn}' for tn in wb_transform_names]
    wild_bootstrap = Aug(dataset_lvl_aug=_wild_bootstrap)
    wb_transforms = [Aug(obs_lvl_aug=_wb_transform) for _wb_transform in _wb_transforms]
    wb_dabstats_over_transforms = [ 
        DABStats(
            stat=lambda xs, key: mmd_with_rbf_kernel(xs, key),
            aug=wild_bootstrap.compose(wb_transform),
            across_dataset_mode='iid'
        )   
        for wb_transform in wb_transforms
    ]

    cis_gen_list = [
        make_cis_gen(make_dab_ci_func(dabstats, cfg.quantile_method, cfg.one_sided), mmd_with_rbf_kernel, WB_post_process, cfg.batch_size)
        for dabstats in wb_dabstats_over_transforms
    ]

    
    assert len(cis_name_list) == len(cis_gen_list)

    return run_CIs(cfg, data_gen, cis_gen_list, cis_name_list)


# MNIST, RBF kernel

def mnist_rbf_kernel(cfg: DABConfig) -> Dict[str, Dict[str, jnp.ndarray]]:
    """simulations"""

    # data
    folder = "data/mnist"
    image_train = np.load(f"{folder}/image_train.npy") / 255.
    label_train = np.load(f"{folder}/label_train.npy")
    image_test = np.load(f"{folder}/image_test.npy") / 255.
    label_test = np.load(f"{folder}/label_test.npy")

    MNIST_DOWNSIZE = 14

    image_train_resized = np.array([resize(img, (MNIST_DOWNSIZE, MNIST_DOWNSIZE), anti_aliasing=True) for img in image_train])
    image_test_resized = np.array([resize(img, (MNIST_DOWNSIZE, MNIST_DOWNSIZE), anti_aliasing=True) for img in image_test])

    n_train = image_train.shape[0]
    image_train_flat = image_train_resized.reshape(n_train, -1)

    n_test = image_test.shape[0]
    image_test_flat = image_test_resized.reshape(n_test, -1)

    image_bandwidth = get_bandwidth(image_test_flat)

    # data setting
    d = cfg.d
    assert d == MNIST_DOWNSIZE * MNIST_DOWNSIZE 
    assert isinstance(cfg.testparam, list) and len(cfg.testparam) >= 1

    if cfg.testparam[0] == 'all':
        noise_level = 0
        image0_flat = image_train_flat
        image1_flat = image_train_flat
    elif cfg.testparam[0] == 'oddeven':
        noise_level = 0
        image0_flat = image_train_flat[jnp.isin(label_train, jnp.array([1,3,5,7,9]))]
        image1_flat = image_train_flat[jnp.isin(label_train, jnp.array([0,2,4,6,8]))]
    elif isinstance(cfg.testparam[0], list):
        if cfg.testparam[0][0] == 'all_noise':
            noise_level = float(cfg.testparam[0][1])
            image0_flat = image_train_flat
            image1_flat = image_train_flat
        else:
            raise ValueError(f"first item of cfg.testparam[0] needs to be one of 'all_noise', but '{cfg.testparam[0][0]}' found.")
    else:
        raise ValueError(f"first item of cfg.testparam needs to be one of 'all' or 'oddeven' or a list, but '{cfg.testparam[0]}' found.")

    def data_gen(n, key, _unused):
        xkey, ykey, noisekey = jax.random.split(key, num=3)
        xs = jnp.take(
            image0_flat, jax.random.choice(xkey, image0_flat.shape[0], shape=(n,), replace=False), axis=0
        )
        ys = jnp.take(
            image1_flat, jax.random.choice(ykey, image1_flat.shape[0], shape=(n,), replace=False), axis=0
        )
        ys += jax.random.normal(noisekey, ys.shape) * noise_level
        return concat_xs_ys_ws(xs, ys, jnp.ones((n,1)))
    
    # specify stats
    rbf_kernel = make_mmd_kernel(
        kernel=make_rbf(image_bandwidth**2)
    )
    mmd_with_rbf_kernel = make_mmd(
        mmd_kernel=rbf_kernel,
        batch_size=cfg.data_batch_size
    )

    # read transforms
    if len(cfg.testparam) > 1:
        rot_angle = float(cfg.testparam[1])
        _random_rotate = _make_random_rotate(-rot_angle, rot_angle)
    else:
        _random_rotate = _make_random_rotate()
    _random_zoom = _make_random_zoom()
    _random_rotate_zoom = _make_random_rotate_zoom()


    wb_transform_names = cfg.compose_transform_names
    _wb_transforms = []
    for tn in wb_transform_names:
        if tn == 'id':
            _wb_transforms.append(_id)
        elif tn == 'zoom':
            _wb_transforms.append(_double_obs_with_weight(_random_zoom))
        elif tn == 'rotate':
            _wb_transforms.append(_double_obs_with_weight(_random_rotate))
        elif tn == 'rotate_zoom':
            _wb_transforms.append(_double_obs_with_weight(_random_rotate_zoom))
        else:
            raise ValueError(f"{tn} is not a valid transformation name")

    # wild bootstrap DAB-compose transforms
    cis_name_list = [f'WB_compose_{tn}' for tn in wb_transform_names]
    wild_bootstrap = Aug(dataset_lvl_aug=_wild_bootstrap)
    wb_transforms = [Aug(obs_lvl_aug=_wb_transform) for _wb_transform in _wb_transforms]
    wb_dabstats_over_transforms = [ 
        DABStats(
            stat=lambda xs, key: mmd_with_rbf_kernel(xs, key),
            aug=wild_bootstrap.compose(wb_transform),
            across_dataset_mode='iid'
        )   
        for wb_transform in wb_transforms
    ]

    cis_gen_list = [
        make_cis_gen(make_dab_ci_func(dabstats, cfg.quantile_method, cfg.one_sided), mmd_with_rbf_kernel, WB_post_process, cfg.batch_size)
        for dabstats in wb_dabstats_over_transforms
    ]
    
    assert len(cis_name_list) == len(cis_gen_list)

    return run_CIs(cfg, data_gen, cis_gen_list, cis_name_list)



# CIFAR, RBF kernel

def cifar_rbf_kernel(cfg: DABConfig) -> Dict[str, Dict[str, jnp.ndarray]]:
    """simulations"""

    # data
    folder = "data/cifar10"
    image_train = np.load(f"{folder}/image_train.npy") / 255.
    label_train = np.load(f"{folder}/label_train.npy")
    image_test = np.load(f"{folder}/image_test.npy") / 255.
    label_test = np.load(f"{folder}/label_test.npy")

    CIFAR_DOWNSIZE = 16

    image_train_resized = np.array([resize(img, (CIFAR_DOWNSIZE, CIFAR_DOWNSIZE), anti_aliasing=True) for img in image_train])
    image_test_resized = np.array([resize(img, (CIFAR_DOWNSIZE, CIFAR_DOWNSIZE), anti_aliasing=True) for img in image_test])

    n_train = image_train.shape[0]
    image_train_flat = image_train_resized.reshape(n_train, -1)

    n_test = image_test.shape[0]
    image_test_flat = image_test_resized.reshape(n_test, -1)

    image_bandwidth = get_bandwidth(image_test_flat)

    # data setting
    d = cfg.d
    assert d == CIFAR_DOWNSIZE * CIFAR_DOWNSIZE * 3
    assert isinstance(cfg.testparam, list) and len(cfg.testparam) >= 1

    if cfg.testparam[0] == 'all':
        image0_flat = image_train_flat
        image1_flat = image_train_flat
    elif cfg.testparam[0] == 'oddeven':
        image0_flat = image_train_flat[jnp.isin(label_train, jnp.array([1,3,5,7,9]))]
        image1_flat = image_train_flat[jnp.isin(label_train, jnp.array([0,2,4,6,8]))]
    elif cfg.testparam[0] == 'odd':
        image0_flat = image_train_flat[jnp.isin(label_train, jnp.array([1,3,5,7,9]))]
        image1_flat = image_train_flat[jnp.isin(label_train, jnp.array([1,3,5,7,9]))]
    else:
        raise ValueError(f"first item of cfg.testparam needs to be one of 'all' or 'oddeven', but '{cfg.testparam[0]}' found.")

    def data_gen(n, key, _unused):
        xkey, ykey = jax.random.split(key)
        xs = jnp.take(
            image0_flat, jax.random.choice(xkey, image0_flat.shape[0], shape=(n,), replace=False), axis=0
        )
        ys = jnp.take(
            image1_flat, jax.random.choice(ykey, image1_flat.shape[0], shape=(n,), replace=False), axis=0
        )
        return concat_xs_ys_ws(xs, ys, jnp.ones((n,1)))
    
    # specify stats
    rbf_kernel = make_mmd_kernel(
        kernel=make_rbf(image_bandwidth**2)
    )
    mmd_with_rbf_kernel = make_mmd(
        mmd_kernel=rbf_kernel,
        batch_size=cfg.data_batch_size
    )

    # read transforms
    if len(cfg.testparam) >= 2:
        assert len(cfg.testparam) == 3
        rot_angle = float(cfg.testparam[1])
        _random_rotate = _make_random_rotate(-rot_angle, rot_angle, num_channels=3)

        zoom_eps = float(cfg.testparam[2])
        _random_zoom = _make_random_zoom(1. - zoom_eps, 1. + zoom_eps, num_channels=3)

        _random_rotate_zoom = _make_random_rotate_zoom(-rot_angle, rot_angle, 1. - zoom_eps, 1. + zoom_eps, num_channels=3)
    else:
        _random_rotate = _make_random_rotate(num_channels=3)
        _random_zoom = _make_random_zoom(num_channels=3)
        _random_rotate_zoom = _make_random_rotate_zoom(num_channels=3)


    wb_transform_names = cfg.compose_transform_names
    _wb_transforms = []
    for tn in wb_transform_names:
        if tn == 'id':
            _wb_transforms.append(_id)
        elif tn == 'zoom':
            _wb_transforms.append(_double_obs_with_weight(_random_zoom))
        elif tn == 'rotate':
            _wb_transforms.append(_double_obs_with_weight(_random_rotate))
        elif tn == 'rotate_zoom':
            _wb_transforms.append(_double_obs_with_weight(_random_rotate_zoom))
        else:
            raise ValueError(f"{tn} is not a valid transformation name")
    

    # wild bootstrap DAB-compose transforms
    cis_name_list = [f'WB_compose_{tn}' for tn in wb_transform_names]
    wild_bootstrap = Aug(dataset_lvl_aug=_wild_bootstrap)
    wb_transforms = [Aug(obs_lvl_aug=_wb_transform) for _wb_transform in _wb_transforms]
    wb_dabstats_over_transforms = [ 
        DABStats(
            stat=lambda xs, key: mmd_with_rbf_kernel(xs, key),
            aug=wild_bootstrap.compose(wb_transform),
            across_dataset_mode='iid'
        )
        for wb_transform in wb_transforms
    ]

    cis_gen_list = [
        make_cis_gen(make_dab_ci_func(dabstats, cfg.quantile_method, cfg.one_sided), mmd_with_rbf_kernel, WB_post_process, cfg.batch_size)
        for dabstats in wb_dabstats_over_transforms
    ]

    assert len(cis_name_list) == len(cis_gen_list)

    return run_CIs(cfg, data_gen, cis_gen_list, cis_name_list)



# HIGGS, RBF kernel

def higgs_rbf_kernel(
    cfg: DABConfig
) -> Dict[str, Dict[str, jnp.ndarray]]:
    """simulations"""
    assert len(cfg.testparam) in [1, 2]

    # HIGGS MMD test
    with open('data/higgs_npy_batches/higgs_batch_0000.npy', 'rb') as f:
        train_data = np.load(f)
    with open('data/higgs_npy_batches/higgs_batch_0001.npy', 'rb') as f:
        test_data = np.load(f)

    higgs_bandwidth = get_bandwidth(test_data)

    # discard class and discard the 7 synthetic features
    train_data_class0 = train_data[train_data[:,0]==0][:, 1:-7]
    train_data_class1 = train_data[train_data[:,0]==1][:, 1:-7]

    # data setting
    assert cfg.testparam[0] in ['SIGNAL', 'BACKGROUND', 'BOTH', 'SIGNAL-BACKGROUND']
    if cfg.testparam[0] == 'SIGNAL':
        xs_source = train_data_class1
        ys_source = train_data_class1
    elif cfg.testparam[0] == 'BACKGROUND':        
        xs_source = train_data_class0
        ys_source = train_data_class0
    elif cfg.testparam[0] == 'BOTH':
        xs_source = jnp.concatenate([train_data_class0, train_data_class1], axis=0)
        ys_source = jnp.concatenate([train_data_class0, train_data_class1], axis=0)
    else:
        xs_source = train_data_class0
        ys_source = train_data_class1

    assert cfg.d == 21

    def data_gen(n, key, _unused):
        xkey, ykey = jax.random.split(key)
        xs = jnp.take(
            xs_source, jax.random.choice(xkey, xs_source.shape[0], shape=(n,), replace=False), axis=0
        )
        ys = jnp.take(
            ys_source, jax.random.choice(ykey, ys_source.shape[0], shape=(n,), replace=False), axis=0
        )
        return concat_xs_ys_ws(xs, ys, jnp.ones((n,1)))

    # statistic
    rbf_kernel = make_mmd_kernel(
        kernel=make_rbf(higgs_bandwidth**2)
    )
    mmd_with_rbf_kernel = make_mmd(
        mmd_kernel=rbf_kernel,
        batch_size=cfg.data_batch_size
    )

    # read transforms
    if len(cfg.testparam) == 2:
        rot_angle = float(cfg.testparam[1])
        _random_rotate = _make_higgs_rotate(-rot_angle, rot_angle)
    else:
        _random_rotate = _make_higgs_rotate()

    wb_transform_names = cfg.compose_transform_names
    _wb_transforms = []
    for tn in wb_transform_names:
        if tn == 'id':
            _wb_transforms.append(_id)
        elif tn == 'random_rotate':
            _wb_transforms.append(_double_obs_with_weight(_random_rotate))
        else:
            raise ValueError(f"{tn} is not a valid transformation name")

    # wild bootstrap DAB-compose transforms
    cis_name_list = [f'WB_compose_{tn}' for tn in wb_transform_names]
    wild_bootstrap = Aug(dataset_lvl_aug=_wild_bootstrap)
    wb_transforms = [Aug(obs_lvl_aug=_wb_transform) for _wb_transform in _wb_transforms]
    wb_dabstats_over_transforms = [
        DABStats(
            stat=lambda xs, key: mmd_with_rbf_kernel(xs, key),
            aug=wild_bootstrap.compose(wb_transform),
            across_dataset_mode='iid'
        )
        for wb_transform in wb_transforms
    ]
    cis_gen_list = [
        make_cis_gen(
            make_dab_ci_func(dabstats, cfg.quantile_method, cfg.one_sided),
            mmd_with_rbf_kernel,
            WB_post_process,
            cfg.batch_size,
        )
        for dabstats in wb_dabstats_over_transforms
    ]

    assert len(cis_name_list) == len(cis_gen_list)
    return run_CIs(cfg, data_gen, cis_gen_list, cis_name_list)
