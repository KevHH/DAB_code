from dataclasses import dataclass
import logging
import os
from typing import Dict, Sequence, List
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy
import pickle

from utils.augs import Aug, _bootstrap, _id, _random_ortho, _rotate_elec
from utils.dabstats import DABStats, DABConfig, run_CIs, make_cis_gen, make_dab_ci_func

"""
    Helper functions
"""

def bootstrap_post_process(lower, upper, emp_stat):
    """
        Bootstrap CI (lower, upper) is for emp_stat - truth, so we transform it into a CI for truth directly
    """
    return (emp_stat - upper, emp_stat - lower)


def bootstrap_no_centring_post_process(lower, upper, emp_stat):
    """
        Bootstrap CI (lower, upper) is for 2*emp_stat - truth, so we transform it into a CI for truth directly
    """
    return (2 * emp_stat - upper, 2 * emp_stat - lower)



"""
    Main functions
"""

# Gaussian and Rademacher and centred Gamma
def simulate_average(cfg: DABConfig, setting: str) -> Dict[str, Dict[str, jnp.ndarray]]:
    """simulations"""
    # specify data and stats
    if setting == 'Gaussian':
        data_gen = lambda n, key, _unused: jax.random.normal(key, shape=(n,cfg.d))
    elif setting == 'Rademacher':
        data_gen = lambda n, key, _unused: jax.random.rademacher(key, shape=(n,cfg.d))
    elif setting == 'CentredGamma':
        data_gen = lambda n, key, _unused: jax.random.gamma(key, a=1, shape=(n,cfg.d)) - 1
    else:
        raise ValueError(f'setting can only be "Gaussian" or "Rademacher", but {setting} found')

    feature = lambda x: jnp.sum(x)
    est = lambda xs: jax.vmap(lambda x: feature(x))(xs)
    ave = lambda xs: jnp.mean(est(xs))
    ste = lambda xs: jnp.std(est(xs)) / jnp.sqrt(xs.shape[0])
    
    # read transforms
    compose_transform_names = list(set(cfg.compose_transform_names))
    compose_transform_fns = []
    for tn in compose_transform_names:
        if tn == 'id':
            compose_transform_fns.append(_id)
        elif tn == 'random_ortho':
            compose_transform_fns.append(_random_ortho)
        else:
            raise ValueError(f"{tn} is not a valid transformation name")
        
    additional_transform_names = list(set(cfg.additional_transform_names))
    additional_transform_fns = []
    for tn in additional_transform_names:
        if tn == 'random_ortho':
            additional_transform_fns.append(_random_ortho)
        else:
            raise ValueError(f"{tn} is not a valid transformation name")
        
    # bootstrap DAB-compose transforms
    cis_name_list = [f'bootstrap_compose_{tn}' for tn in compose_transform_names]
    bootstrap = Aug(dataset_lvl_aug=_bootstrap)
    compose_transforms = [Aug(obs_lvl_aug=_compose_transform) for _compose_transform in compose_transform_fns]
    dabstats_over_transforms = [ 
        DABStats(
            stat=lambda xs, key: ave(xs),
            aug=bootstrap.compose(compose_transform),
            across_dataset_mode='iid',
        )   
        for compose_transform in compose_transforms
    ]

    # add transforms without bootstrap for comparison
    cis_name_list += [f'{tn}' for tn in additional_transform_names]
    transforms = [Aug(obs_lvl_aug=_transform) for _transform in additional_transform_fns]
    dabstats_over_transforms += [ 
        DABStats(
            stat=lambda xs, key: ave(xs),
            aug=transform,
            across_dataset_mode='iid',
        )   
        for transform in transforms
    ]

    cis_gen_list = [
        make_cis_gen(make_dab_ci_func(dabstats, cfg.quantile_method, cfg.one_sided), lambda xs, key: ave(xs), bootstrap_post_process)
        for dabstats in dabstats_over_transforms
    ]

    # Gaussian CI generation
    cis_name_list += ['gaussian']

    def gaussian_ci_func(xs, key, _k, _alpha): 
        upper = jax.scipy.stats.norm.ppf(1-_alpha/2) * ste(xs)
        return (-upper, upper)
    
    cis_gen_list += [ make_cis_gen(gaussian_ci_func, lambda xs, key: ave(xs), bootstrap_post_process) ]

    return run_CIs(cfg, data_gen, cis_gen_list, cis_name_list)


# FermiNet

fname = 'data/ferminet_samples/lithium/qmcjax_ckpt_099999__samples.npz'
data = np.load('data/ferminet_samples/lithium/qmcjax_ckpt_099999__samples.npz')
pos_data = data['positions'][0]


def run_ferminet(cfg: DABConfig) -> Dict[str, Dict[str, jnp.ndarray]]:
    """trained deepsolid network samples"""
    # load data 
    assert cfg.d == 9 # 3 electrons
    d = cfg.d

    # data = jax.random.permutation(jax.random.PRNGKey(1), data, axis=0)
    
    # data_gen = lambda n, _unused, idx: jax.lax.dynamic_slice(pos_data, [n*idx,0], [n,d])

    data_gen = lambda n, key, _unused: jnp.take(
        pos_data, jax.random.choice(key, pos_data.shape[0], shape=(n,), replace=False), axis=0
    )

    # specify stats
    feature = lambda x: x[0] + x[3] + x[6] # x component of the electron dipole moment
    est = lambda xs: jax.vmap(lambda x: feature(x))(xs)
    ave = lambda xs: jnp.mean(est(xs))
    ste = lambda xs: jnp.std(est(xs)) / jnp.sqrt(xs.shape[0])
    
    # read transforms
    compose_transform_names = list(set(cfg.compose_transform_names))
    compose_transform_fns = []
    for tn in compose_transform_names:
        if tn == 'id':
            compose_transform_fns.append(_id)
        elif tn == 'rotate':
            compose_transform_fns.append(_rotate_elec)
        else:
            raise ValueError(f"{tn} is not a valid transformation name")
    
    additional_transform_names = list(set(cfg.additional_transform_names))
    additional_transform_fns = []
    for tn in additional_transform_names:
        if tn == 'rotate':
            additional_transform_fns.append(_rotate_elec)
        else:
            raise ValueError(f"{tn} is not a valid transformation name")
        
    # bootstrap DAB-compose transforms
    cis_name_list = [f'bootstrap_compose_{tn}' for tn in compose_transform_names]
    bootstrap = Aug(dataset_lvl_aug=_bootstrap)
    compose_transforms = [Aug(obs_lvl_aug=_compose_transform) for _compose_transform in compose_transform_fns]
    dabstats_over_transforms = [ 
        DABStats(
            stat=lambda xs, key: ave(xs),
            aug=bootstrap.compose(compose_transform),
            across_dataset_mode='iid',
        )   
        for compose_transform in compose_transforms
    ]

    # add transforms without bootstrap for comparison
    cis_name_list += [f'{tn}' for tn in additional_transform_names]
    transforms = [Aug(obs_lvl_aug=_transform) for _transform in additional_transform_fns]
    dabstats_over_transforms += [ 
        DABStats(
            stat=lambda xs, key: ave(xs),
            aug=transform,
            across_dataset_mode='iid',
        )   
        for transform in transforms
    ]

    cis_gen_list = [
        make_cis_gen(make_dab_ci_func(dabstats, cfg.quantile_method, cfg.one_sided), lambda xs, key: ave(xs), bootstrap_post_process)
        for dabstats in dabstats_over_transforms
    ]

    # Gaussian CI generation
    cis_name_list += ['gaussian']

    def gaussian_ci_func(xs, key, _k, _alpha): 
        upper = scipy.stats.norm.ppf(1-_alpha/2) * ste(xs)
        return (-upper, upper)
    
    cis_gen_list += [ make_cis_gen(gaussian_ci_func, lambda xs, key: ave(xs), bootstrap_post_process) ]

    return run_CIs(cfg, data_gen, cis_gen_list, cis_name_list)

