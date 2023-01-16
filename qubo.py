from typing import Union

import numpy as np
from numpy.random import RandomState
from numpy.typing import ArrayLike

from misc import get_random_state


#
# Methods for creating QUBO instances
#

def random_subset_sum(n: int, value_range=(-1.0, 1.0), subset_size: int=None, random_state: Union[RandomState, int]=None):
    """Creates QUBO instance of a random Subset Sum problem.

    Args:
        n (int): Number of values (QUBO dimension).
        value_range (tuple, optional): Minimum and maximum value. Defaults to (-1.0, 1.0).
        subset_size (int, optional): Size of optimal subset. If None (default), the subset is sampled uniformly.
        random_state (Union[RandomState, int], optional): Random state or seed for reproducibility. Defaults to None.

    Returns:
        ArrayLike: upper-triangular QUBO parameter matrix of shape (n, n).
        ArrayLike: binary mask of shape (n,), indicating the global optimum
    """
    npr = get_random_state(random_state)
    values = np.hstack((npr.uniform(*value_range, size=n-2), *value_range))
    npr.shuffle(values)
    if subset_size is None:
        while True:
            subset = npr.binomial(1, 0.5, size=n).astype(bool)
            if subset.sum() > 0:
                # ensure that subset has at least size 1
                break
    else:
        subset = npr.permutation(n) < subset_size

    target = values[subset].sum()
    qubo = np.outer(values, values) - 2*target*np.diag(values)
    qubo = np.triu(qubo) + np.tril(qubo, -1).T # make upper triangle
    return qubo, subset


#
# Methods for modifying QUBO instances
#

def scale(x: ArrayLike, max_norm: float=1.0) -> ArrayLike:
    """Scale array to match given max norm.

    Args:
        x (ArrayLike): Input array.
        max_norm (float, optional): New max norm. Defaults to 1.0.

    Returns:
        ArrayLike: Scaled array.
        float: Scale factor applied to x.
    """
    factor = max_norm/np.abs(x).max()
    return x * factor, factor


def to_bit_range(x: ArrayLike, bits: int=16) -> ArrayLike:
    """Round array to fit within given bit range.

    Args:
        x (ArrayLike): Input array.
        bits (int, optional): Number of bits (signed). Defaults to 16.

    Returns:
        ArrayLike: Rounded array. Note this will still be of floating type.
    """
    assert bits <= 64, 'Cannot represent bit ranges beyond 64 bits'
    scaled, factor = scale(x, max_norm=2**(bits-1)-1)
    return scaled.round()/factor


def to_sparse_dict(x: ArrayLike):
    n = x.shape[0]
    ixs = np.triu_indices_from(x)
    d = { ix: v for ix, v in zip(zip(*ixs), x[ixs]) if not np.isclose(v, 0) }
    # make sure that all variables are present!
    all_vars = { i for t in d.keys() for i in t }
    for i in set(range(n)).difference(all_vars):
        d[(i, i)]=0
    return d
