import functools

import jax
import jax.numpy as jnp

def batch_keys_from_seed(seed: int, num_sim: int):
    return jax.random.split(jax.random.PRNGKey(seed), num_sim)

def batch_keys_split(keys: jax.Array, num: int = 2):
    return jax.vmap(lambda key: jax.random.split(key, num))(keys).swapaxes(0,1)


def batched_fori_loop(batch_size: int = None):
    """Decorator that applies a batched function over axis 0 via jax.lax.fori_loop."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if batch_size is None:
                return fn(*args, **kwargs)
            if batch_size <= 0:
                raise ValueError(f"batch_size must be positive, but got {batch_size}")
            if len(args) == 0:
                raise ValueError("batched_fori_loop expects at least one positional array argument")

            num_items = args[0].shape[0]
            num_batches = (num_items + batch_size - 1) // batch_size
            padded_num_items = num_batches * batch_size
            pad_len = padded_num_items - num_items

            padded_args = tuple(
                jnp.pad(arg, [(0, pad_len)] + [(0, 0)] * (arg.ndim - 1))
                for arg in args
            )
            batched_args = tuple(
                arg.reshape((num_batches, batch_size) + arg.shape[1:])
                for arg in padded_args
            )

            first_batch_args = tuple(arg[0] for arg in batched_args)
            first_batch_out = fn(*first_batch_args, **kwargs)
            out_init = jax.tree_util.tree_map(
                lambda x: jnp.zeros((padded_num_items,) + x.shape[1:], dtype=x.dtype),
                first_batch_out
            )
            out_init = jax.tree_util.tree_map(
                lambda out, batch: jax.lax.dynamic_update_slice(
                    out,
                    batch,
                    (0,) + (0,) * (out.ndim - 1),
                ),
                out_init,
                first_batch_out,
            )

            def _batch_body(batch_idx, out_carry):
                batch_args = tuple(
                    jax.lax.dynamic_index_in_dim(arg, batch_idx, axis=0, keepdims=False)
                    for arg in batched_args
                )
                batch_out = fn(*batch_args, **kwargs)
                return jax.tree_util.tree_map(
                    lambda out, batch: jax.lax.dynamic_update_slice(
                        out,
                        batch,
                        (batch_idx * batch_size,) + (0,) * (out.ndim - 1),
                    ),
                    out_carry,
                    batch_out,
                )

            out = jax.lax.fori_loop(1, num_batches, _batch_body, out_init)
            return jax.tree_util.tree_map(lambda x: x[:num_items], out)

        return wrapped

    return decorator


"""
    utils for wild bootstrap
"""

def concat_xs_ys_ws(xs: jnp.ndarray, ys: jnp.ndarray, ws: jnp.ndarray):
    """
    :param xs: shape (n,d), representing dataset 1
    :param ys: shape (n,d), representing dataset 2
    :param ws: shape (n,1), representing weights
    """
    assert xs.shape[0] == ys.shape[0] == ws.shape[0]
    assert len(xs.shape) == len(ys.shape) == len(ws.shape) == 2
    return jnp.concat([xs, ys, ws], axis=1)

def split_xs_ys_ws(xs_ys_ws: jnp.ndarray):
    """
    :param xs_ys_ws: shape (n,2*d+1)
    """
    assert (xs_ys_ws.shape[1] - 1) % 2 == 0
    d = (xs_ys_ws.shape[1] - 1) // 2 
    return xs_ys_ws[:,:d], xs_ys_ws[:,d:(2*d)], xs_ys_ws[:,-1:]

