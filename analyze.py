from itertools import combinations
from operator  import attrgetter, itemgetter

import bitvec
import numpy as np
from kneed             import KneeLocator
from scipy.stats       import wasserstein_distance
from sklearn.neighbors import NearestNeighbors


#
# Methods for analyzing annealing results
#

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


#
# Methods for analyzing QUBO instances
#

def dynamic_range(qubo, bits=False):
    params = np.r_[qubo[np.triu_indices_from(qubo, 1)], 0]
    dmin, dmax = minmax(abs(u-v) for u, v in combinations(params, r=2) if not np.isclose(u, v))
    return np.log2(dmax/dmin) if bits else 20*np.log10(dmax/dmin)


def epsilon_for_dbscan(qubo, k=5):
    params = qubo[np.triu_indices_from(qubo)][:, np.newaxis]
    knn = NearestNeighbors(n_neighbors=k)
    fit = knn.fit(params)
    distances, _ = fit.kneighbors(params)
    distances = np.sort(distances[:, 1:].mean(1))
    kneeloc = KneeLocator(np.arange(distances.size), distances, curve='convex', direction='increasing')
    return kneeloc.knee_y


def induced_ordering(qubo):
    n = qubo.shape[0]
    values = [ x @ qubo @ x for x in bitvec.all(n) ]
    return np.argsort(values)

#
# Random miscellaneous methods
#

def minmax(it):
    xmin = float('inf')
    xmax = float('-inf')
    for x in it:
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
    return xmin, xmax


if __name__ == '__main__':
    from qubo import random_subset_sum

    qubo, _ = random_subset_sum(64)
    res = epsilon_for_dbscan(qubo)
