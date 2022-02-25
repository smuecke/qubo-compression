from genericpath import sameopenfile
from operator import attrgetter, itemgetter

import numpy as np
from scipy.stats import wasserstein_distance


def dict_to_vec(sample, n):
    full = { i: 0 for i in range(n) }
    full.update(sample)
    return 


def energy_arrays(result, qubo):
    arr_reported = []
    arr_true = []
    for sample, reported_energy, occ, *_ in sorted(result.aggregate().record, key=itemgetter(1)):
        true_energy = sample @ qubo @ sample
        arr_true.extend([true_energy]*occ)
        arr_reported.extend([reported_energy]*occ)
    return np.vstack((
        np.asarray(arr_reported, dtype=np.float64),
        np.asarray(arr_true, dtype=np.float64)))


def wasserstein(result, qubo):
    """Calculates the Wasserstein distance between the energy distribution reported
       by the sampler and the actual energy values of the original QUBO instance"""
    sampler_energies = []
    true_energies    = []
    weights = []
    for sample in sorted(result.aggregate().data(), key=attrgetter('energy')):
        sampler_energies.append(sample.energy)
        x = np.fromiter(sample.sample.values(), dtype=np.float64)
        true_energies.append(x @ qubo @ x)
        weights.append(sample.num_occurrences)
    return wasserstein_distance(sampler_energies, true_energies, weights, weights)