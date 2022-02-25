# QUBO Compression

Research project for investigating the effects of compression and rounding on the energy landscape of QUBO instances.

## Requirements

This project requires the D-Wave software packages `dimod`, `dwave_system` and `dwave_neal`.
For performing experiments on quantum hardware, you need a D-Wave API token.
See `requirements.txt` for detailed package requirements.


## How to run

```
main.py [-h] [-s SEED] [-o OUTPUT] [--simulate] n

positional arguments:
  n                     Number of qubits

options:
  -h, --help            show this help message and exit
  -s SEED, --seed SEED  Random seed for reproducibility
  -o OUTPUT, --output OUTPUT
                        Output directory for experiment results
  --simulate            Use tabu search instead of QPU
```
