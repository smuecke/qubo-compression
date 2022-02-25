import json
from datetime import datetime
from hashlib  import md5
from typing   import Union

import numpy as np
from numpy.random import RandomState
from numpy.typing import ArrayLike


def get_random_state(random_state: Union[int, RandomState]):
    """Either return random state or create random state from seed"""
    if isinstance(random_state, RandomState):
        return random_state
    else:
        # int or None -> use as seed
        return RandomState(random_state)


def get_numerical_seed(seed: Union[int, str]):
    if seed is None:
        return None
    try:
        seed_ = int(seed)
    except ValueError:
        seed_ = int(md5(seed.encode('utf-8')).hexdigest(), 16) & 0xffffffff


def random_seeds(random_state: Union[int, RandomState]):
    npr = get_random_state(random_state)
    while True:
        yield npr.randint(2**32)


def timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)