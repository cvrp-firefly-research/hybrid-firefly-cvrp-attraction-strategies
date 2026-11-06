# Hybrid Firefly Algorithm for the CVRP

This repository contains the Python source code associated with the research article:

**"Attraction Strategies in a Hybrid Firefly Algorithm for the Capacitated Vehicle Routing Problem"**

Submitted to *Revista Científica*, Universidad Distrital Francisco José de Caldas.

## Attraction Strategies

The implementation includes three attraction strategies:

- **BSF:** Best-So-Far Attraction.
- **PrevBest:** Previous-Iteration Best Attraction.
- **PairBE:** Pairwise Better-or-Equal Firefly Attraction.

## Installation

Python 3 is required.

Install the dependencies using:

```bash
pip install -r requirements_full.txt
```

## Execution

Run the experiments from the project root directory:

```bash
python experiments.py
```

The experiment configuration is defined in `experiments.py`, including the instance directory, attraction strategies, population size, number of iterations, and repetitions.

## Experimental Configuration

- Population size: 25 fireflies
- Maximum iterations: 1,000
- Independent repetitions: 10
- Crossover probability: 0.95
- Mutation probability: 0.95

The experiments use CVRP instances in `.vrp` format. The input directory is specified in `experiments.py`.

## Code Availability

The source code is provided to support scientific review and reproducibility of the accompanying research article.

**All rights reserved.** No open-source license has been granted at this stage.
