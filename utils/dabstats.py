from dataclasses import dataclass
from typing import Sequence, Dict, List
import logging
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from utils.augs import Aug, _iid_dataset_lvl, _enumerate_dataset_lvl
from utils.utils import batch_keys_from_seed, batch_keys_split, batched_fori_loop

class DABStats:
    """
        Statistic class

        :var stat: function (xs: jnp.ndarray, key: jax.Array ) -> jnp.ndarray representing the statistic to compute over a dataset xs, potentially random (with randomness specified by key)
        :var aug: List[Aug] objects, representing list of augmentations
        :var across_dataset_mode: specify how to transform at the dataset level. either 'iid' or 'enum'.
        :var aug_new_to_average: optional Aug object. If specified, use DAB-average rather than DAB-compose, and average over aug_new_to_average first
    """
    def __init__(self, stat, aug: Aug | List[Aug], across_dataset_mode: str):
        self.stat = stat
        
        if across_dataset_mode == 'iid':
            def get_aug_stats(xs: jnp.ndarray, key: jax.Array, num: int):
                stat_key, key = jax.random.split(key)
                fixed_stat = partial(stat, key=stat_key)
                return _iid_dataset_lvl(
                    lambda xs, key: fixed_stat(aug.dataset_lvl_aug(xs, key))
                )(xs, key, num)
        elif across_dataset_mode == 'enum':
            def get_aug_stats(xs: jnp.ndarray, key: jax.Array, num: int):
                stat_key, key = jax.random.split(key)
                fixed_stat = partial(stat, key=stat_key)
                return _enumerate_dataset_lvl(
                    [
                        lambda xs, key, aug_single=aug_single: fixed_stat(aug_single.dataset_lvl_aug(xs, key))
                        for aug_single in aug
                    ]
                )(xs, key, num)
        else:
            raise ValueError(f"across_dataset_mode can only be either 'iid' or 'enum' but {across_dataset_mode} found.")

        self.get_aug_stats = get_aug_stats
        self.jitted_get_aug_stats = jax.jit(self.get_aug_stats, static_argnums=2)

    def get_CI(self, 
                   xs: jnp.ndarray,
                   num: int,
                   key: jax.Array,
                   alpha: float = 0.05,
                   quantile_method: str = 'linear',
                   one_sided: str = None,
        ):
        """
            Get DAB confidence interval
        
            :param xs: (n,...) data
            :param num: number of dataset-level augmentations
            :param key: jax.Array representing a key
            :param alpha: target coverage probability
            :param quantile_method: how to give CI quantile
            :parma one_sided: None (two-sided interval is used), 'upper' or 'lower
            :
            :return (lower, upper, ave): lower and upper values of the CI as well as the average stat over all augmented stats
        """
        aug_stats = self.jitted_get_aug_stats(xs, key, num=num)
        # aug_stats = self.get_aug_stats(xs, key, num=num)
        if one_sided is None:
            lower_cutoff = alpha/2
            upper_cutoff = 1. - alpha/2
        elif one_sided == 'upper':
            lower_cutoff = 0.
            upper_cutoff = 1. - alpha
        elif one_sided == 'lower':
            lower_cutoff = alpha
            upper_cutoff = 1.
        else:
            raise ValueError(f"one_sided can only be None, 'upper' or 'lower', but {one_sided} found") 

        return ( jnp.quantile(aug_stats, lower_cutoff, method=quantile_method),
                    jnp.quantile(aug_stats, upper_cutoff, method=quantile_method))



@dataclass
class DABConfig:
    """
        :param stat: (xs: jnp.ndarray) -> float, function that gives the 1d statistic of a dataset
        :param ste: (xs: jnp.ndarray) -> float, function that gives the standard error estimate of the 1d statistic of a dataset
        :param data_gen: (n: int, key: jax.Array) -> jnp.ndarray, function that generates a dataset of n points with given jax key
    """
    n: Sequence[int]
    num_transform: Sequence[int]
    compose_transform_names: Sequence[str]
    label: str
    d: int = 0
    seed: int = 0
    num_sim: int = 100
    alpha: Sequence[float] = (0.05,)
    quantile_method: str = 'linear'
    one_sided: str = None
    testparam: object = None #parameter used in testing
    average_transform_names: Sequence[str] = None
    num_average_transforms: int = None
    additional_transform_names: Sequence[str] = None #additional transforms without composing with old transforms
    batch_size: int = None
    data_batch_size: int = None
    

    def get_path(self):
        return (f'logs/{self.label}' 
                + '_n' + (f'{self.n[0]}-{self.n[-1]}' if len(self.n) > 1 else f'{self.n[0]}')
                + f'_d{self.d}'
                + '_k' + (f'{self.num_transform[0]}-{self.num_transform[-1]}' if len(self.num_transform) > 1 else f'{self.num_transform[0]}')
                + '_alpha' + (f'{self.alpha[0]:.2f}-{self.alpha[-1]:.2f}' if len(self.alpha) > 1 else f'{self.alpha[0]:.2f}')
                + (f'_testparam{"".join([str(a) for a in self.testparam])}' if self.testparam is not None else '')
                + f'_seed{self.seed}_sim{self.num_sim}'
                + (f'_qm{self.quantile_method}' if self.quantile_method != 'linear' else '')
                )

def make_dab_ci_func(dabstats, quantile_method='linear', one_sided=None):
    return lambda xs, k, key, alpha: dabstats.get_CI(xs, k, key, alpha, quantile_method, one_sided)


def make_cis_gen(ci_func, stat, post_process=None, batch_size: int = None):
    @partial(jax.jit, static_argnames=("k"))
    def cis_gen(data_sims, keys, k, alpha):
        @batched_fori_loop(batch_size=batch_size)
        def _get_ci_over_batch(xs_batch, key_batch):
            return jax.vmap(
                lambda xs, key: ci_func(xs, k, key, alpha)
            )(xs_batch, key_batch)

        ci_keys, emp_keys = batch_keys_split(keys)
        lower, upper = _get_ci_over_batch(data_sims, ci_keys)
        emp_stat = jax.vmap( lambda xs, key: stat(xs, key) )(data_sims, emp_keys)

        if post_process is not None:
            pp_lower, pp_upper = post_process(lower, upper, emp_stat)
        else:
            pp_lower = lower
            pp_upper = upper
        return jnp.stack([pp_lower, emp_stat, pp_upper])
        
    return cis_gen


def run_CIs(
        cfg: DABConfig, 
        data_gen, 
        cis_gen_list, 
        cis_name_list,
    ) -> Dict[str, Dict[str, jnp.ndarray]]:
    """
    Generate confidence intervals.
    
    :param cfg: Description
    :type cfg: DABConfig
    :param data_gen: Description
    :param cis_gen_list: Description
    :param cis_name_list: Description
    :return: Description
    :rtype: Dict[str, Dict[str, ndarray]]
    """
    sim_keys = batch_keys_from_seed(cfg.seed, cfg.num_sim)

    base_path = cfg.get_path()
    os.makedirs(base_path, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(base_path, "log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logging.info("created output dir=%s", base_path)

    # go through different settings
    for n in cfg.n:
        data_keys, sim_keys = batch_keys_split(sim_keys)
        data_idxs = jnp.arange(cfg.num_sim)

        @batched_fori_loop(batch_size=cfg.batch_size)
        def _generate_data_sims(data_key_batch, data_idx_batch):
            # Batch dataset generation to avoid tracing extremely large
            # replace=False permutations across all simulations at once.
            return jax.vmap(lambda key, idx: data_gen(n, key, idx))(data_key_batch, data_idx_batch)

        data_sims = _generate_data_sims(data_keys, data_idxs)

        for k in cfg.num_transform:
            cis_alpha_list = []

            for alpha in cfg.alpha:
                logging.info(
                    "computing n=%s k=%s alpha=%s",
                    n,
                    k,
                    f"{alpha:.2f}"
                )
                # CI generation 
                cis_list = []
                for cis_gen in cis_gen_list:
                    ci_keys, sim_keys = batch_keys_split(sim_keys)
                    cis_list.append(
                        cis_gen(data_sims, ci_keys, k, alpha)
                    )
                cis_alpha_list.append(cis_list)

            # log and save per (n,k)
            save_path = f"{base_path}/_n{n}_k{k}.npz"
            cis_alpha_list = np.array(cis_alpha_list).swapaxes(0,1) # s.t. axis 0 is over different cis_gens and axis 1 is over alphas
            cis_dict = {
                    f'{tn}_cis': cis.swapaxes(0,1) # s.t. axis 0 is (lower, mean, upper) and axis 1 is over alphas
                        for tn, cis in zip(cis_name_list, cis_alpha_list)
            }
            np.savez_compressed(
                save_path,
                n=n,
                k=k,
                alpha=cfg.alpha,
                testparam=cfg.testparam,
                **cis_dict
            )
    return base_path

