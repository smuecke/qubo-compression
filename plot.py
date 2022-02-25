import json

import matplotlib.pyplot as plt
import numpy             as np
from matplotlib.cm import viridis

from misc import load_json


def plot_results(energies_file: str, meta_file: str, **kwargs):
    energies = np.load(energies_file)
    meta     = load_json(meta_file)

    fig, (ax_left, ax_right)  = plt.subplots(1, 2, sharey='all')
    qubo_ix = kwargs['qubo_ix']
    xs = np.arange(meta['reads'])
    for  b, col in zip(reversed(meta['bits']), viridis(np.linspace(0,1,len(meta['bits'])))):
        ys = energies[f'{b}_bits']
        ax_left.scatter(x=xs, y=ys[qubo_ix, 0, :], s=0.4, color=col, alpha=0.8, label=f'{b} Bits')
        ax_right.scatter(x=xs, y=ys[qubo_ix, 1, :], s=0.4, color=col, alpha=0.8)
    ax_left.hlines(meta['optimal_energies'][qubo_ix], xs.min(), xs.max(), linestyles='dotted', color='black', label='Optimal')
    ax_right.hlines(meta['optimal_energies'][qubo_ix], xs.min(), xs.max(), linestyles='dotted', color='black')

    ax_left.set_title('Reported Energy')
    ax_left.set_xlabel('Shots')
    ax_left.set_ylabel('Energy')
    ax_left.grid(True)
    ax_left.legend(loc='upper left')
    ax_right.set_title('Actual Energy')
    ax_right.set_xlabel('Shots')
    ax_right.grid(True)

    fig.tight_layout()
    fig.savefig('output/plot.pdf')
    #fig.show()