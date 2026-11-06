"""
Experimental runner for the research article:
"Attraction Strategies in a Hybrid Firefly Algorithm
for the Capacitated Vehicle Routing Problem"

Submitted to Revista Científica,
Universidad Distrital Francisco José de Caldas.

Executes the BSF, PrevBest, and PairBE attraction strategies
using the experimental configurations defined in this script.
"""

from utils import main_FireFly_HFA
from utils import CVRProblem
from pathlib import Path

if __name__ == "__main__":
    ff_comp_BSF = 0  # BSF: Best-So-Far Attraction
    ff_comp_PrevBest = 1  # PrevBest: Previous-Iteration Best Attraction
    ff_comp_PairBE = 2  # PairBE: Pairwise Better-or-Equal Firefly Attraction

    # instances_path = Path("./instances/A") # Instance set A
    instances_path = Path("./instances/nov") # Instance A-n32-k5
    dir_result_base = "./result1/resultNOV/"

    instances_files = instances_path.rglob("*.vrp")
    instances_files = [f.__str__() for f in instances_files]
    instances_files.sort()
    problems = [CVRProblem(f) for f in instances_files]

    main_FireFly_HFA(
        problems,
        repetitions=10,
        max_iter=1000,
        pop_size=25,
        crossover_rate=0.95,
        mutation_rate=0.95,
        ff_comp=ff_comp_BSF,
        dir_result_base=dir_result_base,
    )
    main_FireFly_HFA(
        problems,
        repetitions=10,
        max_iter=1000,
        pop_size=25,
        crossover_rate=0.95,
        mutation_rate=0.95,
        ff_comp=ff_comp_PrevBest,
        dir_result_base=dir_result_base,
    )
    main_FireFly_HFA(
        problems,
        repetitions=10,
        max_iter=1000,
        pop_size=25,
        crossover_rate=0.95,
        mutation_rate=0.95,
        ff_comp=ff_comp_PairBE,
        dir_result_base=dir_result_base,
    )
    # main_FireFly_HFA(problems, repetitions=10, max_iter=20, pop_size=15, crossover_rate=0.95, mutation_rate=0.10, ff_comp=ff_comp_PairBE, dir_result_base=dir_result_base)
