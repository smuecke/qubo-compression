import json
from argparse import ArgumentParser
from os       import path

import numpy as np
from dimod        import BinaryQuadraticModel as BQM
from dwave.system import DWaveSampler, EmbeddingComposite
from neal         import SimulatedAnnealingSampler
from tqdm         import trange

from analyze import energy_arrays
from misc    import get_numerical_seed, get_random_state, random_seeds, save_json, timestamp
from qubo    import random_subset_sum, to_bit_range


def get_args():
    p = ArgumentParser()
    p.add_argument('n', type=int, help='Number of qubits')
    p.add_argument('-s', '--seed', default=None, help='Random seed for reproducibility')
    p.add_argument('-o', '--output', default='output/', help='Output directory for experiment results')
    p.add_argument('--simulate', action='store_true', help='Use tabu search instead of QPU')
    return vars(p.parse_args())


def main(n: int, seed=None, output='', simulate=False):
    random_state = get_random_state(get_numerical_seed(seed))
    sampler = SimulatedAnnealingSampler() if simulate else EmbeddingComposite(DWaveSampler())
    qubo, opt = random_subset_sum(n)
    
    # manual settings
    bits        = [32, 16, 8, 4]
    reads       = 1024
    repetitions = 5

    all_energies = np.empty((len(bits), repetitions, 2, reads))
    meta_info    = {
        'n': n,
        'bits': bits,
        'reads': reads,
        'repetitions': repetitions,
        'global_seed': seed,
        'sampler': sampler.__class__.__name__,
        'optimal_energies': []}

    seeds = random_seeds(random_state)
    for qubo_ix in trange(repetitions):
        qubo, opt = random_subset_sum(n, random_state=random_state)
        meta_info['optimal_energies'].append(opt @ qubo @ opt)
        for bit_ix, b in enumerate(bits):
            qubo_ = to_bit_range(qubo, bits=b)
            bqm = BQM(qubo_, 'BINARY')
            additional_kwargs = {'seed': next(seeds)} if simulate else {'label': f'{n=}@{b} bits'}
            result = sampler.sample(bqm, num_reads=reads, **additional_kwargs)
            # record energy levels
            all_energies[bit_ix, qubo_ix, ...] = energy_arrays(result, qubo)

    ts = timestamp()
    save_json(meta_info, path.join(output, f'{n}_{ts}_meta.json'))
    np.savez_compressed(
        path.join(output, f'{n}_{ts}_energies'),
        **{f'{b}_bits': arr for b, arr in zip(bits, all_energies)})
    
if __name__ == '__main__':
    main(**get_args())