"""
Hybrid Firefly Algorithm for the Capacitated Vehicle Routing Problem.

Implements the attraction strategies:
- BSF: Best-So-Far Attraction
- PrevBest: Previous-Iteration Best Attraction
- PairBE: Pairwise Better-or-Equal Firefly Attraction

Associated research article:
"Attraction Strategies in a Hybrid Firefly Algorithm
for the Capacitated Vehicle Routing Problem"
"""

import numpy as np
import numpy.typing as npt
from utils import SolutionHFA
from utils import CVRProblem
import time


class FireFlyHFA:
    def __init__(
        self,
        max_iter: int,
        pop_size: int = 15,
        crossover_rate: float = 0.95,
        mutation_rate: float = 0.1,
        ff_comp: int = 0,
    ):

        self.problem: CVRProblem = None
        self.fireflies = None
        self.intensity = None
        self.function = None
        self.best = None
        self.show = True
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.ff_comp = ff_comp
        self.round = True

    def solve(self, seed: int, problem: CVRProblem, base_name=None):
        total_timer = time.time()
        if seed == 0:
            print(f"\nstart, Problem: {problem} - seed: {seed}.")
        else:
            print(f"start, Problem: {problem} - seed: {seed}.")
        self.problem = problem
        max_efos = 0
        self.my_print(problem)
        np.random.seed(seed)
        best_fitness_history = np.zeros(self.max_iter + 1, dtype=float)

        self.intensity = np.zeros(self.pop_size, dtype=float)
        self.fireflies: np.ndarray[SolutionHFA] = np.ndarray(
            self.pop_size, dtype=np.object_
        )
        next_fireflies = np.empty(self.pop_size, dtype=SolutionHFA)

        self.fireflies[0] = SolutionHFA(self.problem)
        self.fireflies[0].initialization_Sweep1()
        next_fireflies[0] = SolutionHFA(self.problem)
        self.best = SolutionHFA(self.problem)
        self.best.from_solution(self.fireflies[0])
        customers = np.random.choice(
            range(1, self.problem.size), self.problem.size - 1, replace=False
        )
        for i in range(1, self.pop_size):
            self.fireflies[i] = SolutionHFA(self.problem)
            self.fireflies[i].initialization_Sweep1()
            next_fireflies[i] = SolutionHFA(self.problem)
            if self.fireflies[i].fitness < self.best.fitness:
                self.best.from_solution(self.fireflies[i])
        self.my_print(self.best)
        self.show_fitness(self.fireflies)
        self.fireflies = np.sort(self.fireflies)
        self.show_fitness(self.fireflies)
        self.comp_intensity()
        self.my_print(self.intensity)

        num_efo = self.pop_size
        epsilon = 1e-4
        t = 0
        if self.round:
            best_fitness_history[t] = self.best.evaluate_round()
        else:
            best_fitness_history[t] = self.best.fitness
        if base_name:
            self.best.save(f"{base_name}_iter_{t:04d}.sol")
        actual_ffly = SolutionHFA(self.problem)
        bestLast = SolutionHFA(self.problem)
        good_active_ffly = SolutionHFA(self.problem)
        independent_ffly = True
        best_t = 0
        while t < self.max_iter:
            # print('.', end='')
            t += 1

            start_timer = time.time()
            for i in range(self.pop_size):
                good_active_ffly.from_solution(self.fireflies[i])
                next_fireflies[i].from_solution(self.fireflies[i])
                # actual_ffly.from_solution(self.fireflies[i])
                actual_ffly: SolutionHFA = next_fireflies[i]
                bestLast.from_solution(self.best)
                if self.ff_comp == 3:
                    rand_pmx = np.random.randint(3)
                else:
                    rand_pmx = self.ff_comp
                for j in range(i):  # range(self.pop_size):
                    if self.intensity[i] > self.intensity[j]:
                        aux_val_fitness = next_fireflies[i].fitness
                        if np.random.random() < self.crossover_rate:
                            actual_ffly.check()
                            if rand_pmx == 0:
                                actual_ffly.pmx(self.best)
                            elif rand_pmx == 1:
                                actual_ffly.pmx(self.fireflies[0])
                            else:
                                actual_ffly.pmx(self.fireflies[j])

                            actual_ffly.check()
                            actual_ffly.change_one()
                            if actual_ffly.fitness < next_fireflies[i].fitness:
                                next_fireflies[i].from_solution(actual_ffly)
                            if actual_ffly.fitness < good_active_ffly.fitness:
                                good_active_ffly.from_solution(actual_ffly)
                            if actual_ffly.fitness < self.best.fitness:
                                self.best.from_solution(actual_ffly)
                                best_t = t
                                self.best.check()
                        actual_ffly.new_2_opt()
                        actual_ffly.change_one()
                        if actual_ffly.fitness < next_fireflies[i].fitness:
                            next_fireflies[i].from_solution(actual_ffly)
                        if actual_ffly.fitness < good_active_ffly.fitness:
                            good_active_ffly.from_solution(actual_ffly)
                        if actual_ffly.fitness < self.best.fitness:
                            self.best.from_solution(actual_ffly)
                            best_t = t
                            self.best.check()
                        actual_ffly.mutationHFA(self.mutation_rate)
                        actual_ffly.check()
                        if actual_ffly.fitness < next_fireflies[i].fitness:
                            next_fireflies[i].from_solution(actual_ffly)
                        if actual_ffly.fitness < good_active_ffly.fitness:
                            good_active_ffly.from_solution(actual_ffly)
                        if actual_ffly.fitness < self.best.fitness:
                            self.best.from_solution(actual_ffly)
                            best_t = t
                            self.best.check()
                        num_efo += 1
                if (
                    abs(next_fireflies[i].fitness - self.fireflies[i].fitness) > 1e-4
                ):  # OJO actualizar cuando se mueva
                    if next_fireflies[i].fitness < self.fireflies[i].fitness:
                        self.fireflies[i].from_solution(next_fireflies[i])

                if (self.best.fitness < bestLast.fitness) and abs(
                    self.best.fitness - bestLast.fitness
                ) > 1e-4:
                    self.fireflies[i].from_solution(self.best)
                    self.my_print(
                        f"Best  t: {t}, i:{i}, efos: {num_efo}, best: {self.best.fitness:.5f}"
                    )

            if self.round:
                best_fitness_history[t] = self.best.evaluate_round()
            else:
                best_fitness_history[t] = self.best.fitness
            best_fitness_found = best_fitness_history[t]
            end_timer = time.time()
            time_spend = end_timer - start_timer
            self.my_print(
                f"\nPrincipal HFA\nt: {t}, efos: {num_efo}, Time: {time_spend:.5f}"
            )
            self.show_fitness(self.fireflies)
            self.fireflies = np.sort(self.fireflies)
            next_fireflies = np.sort(next_fireflies)
            self.show_fitness(self.fireflies)
            val_min = self.fireflies[0].fitness
            val_min_ff0 = val_min
            for i in range(1, self.pop_size):
                if (
                    abs(val_min - self.fireflies[i].fitness) < 1e-4
                ):  # equal the best firefly
                    if np.random.random() < 0.1:
                        self.fireflies[i].mutationHFA(1)
                    else:
                        self.fireflies[i].init_local()
                val_min = self.fireflies[i].fitness
            self.fireflies = np.sort(self.fireflies)
            self.show_fitness(self.fireflies)
            if abs(val_min_ff0 - self.fireflies[0].fitness) > 1e-4:
                self.best.from_solution(self.fireflies[0])
                if self.round:
                    best_fitness_history[t] = self.best.evaluate_round()
                else:
                    best_fitness_history[t] = self.best.fitness
            self.comp_intensity()
            self.my_print(self.intensity)
            self.best.sort_routes()
            if base_name:
                for i in range(self.pop_size):
                    self.fireflies[i].save(f"{base_name}_iter_{t:04d}_{i}.sol")
            self.my_print(self.best)
            self.my_print(
                f'Principal HFA, t: {t}, efos: {num_efo}, Fitness Round {self.best.evaluate_round()} \n{"-" * 20}'
            )
            if abs(best_fitness_found - self.problem.bestKnown) <= 1e-4:
                self.my_print(f"NEW {self.best.fitness} <= {self.problem.bestKnown}")
                while t < self.max_iter:
                    t += 1
                    best_fitness_history[t] = best_fitness_found
                break
        if seed == 0:
            print(
                f"\nEnd, Problem: {problem} - seed: {seed}. Fitness: {best_fitness_history[-1]} - Iter: {best_t}"
            )
        else:
            print(
                f"End, Problem: {problem} - seed: {seed}. Fitness: {best_fitness_history[-1]} - Iter: {best_t}"
            )

        return (
            best_fitness_history,
            self.best,
            best_t,
            time.time() - total_timer,
            problem.file_name,
            seed,
        )

    def comp_intensity(self):
        for i in range(self.fireflies.size):
            self.fireflies[i].sort_routes()
            self.intensity[i] = self.fireflies[i].hd(self.fireflies[0])
        self.my_print(f"\n{self.intensity}")
        iterable = (
            np.random.randint(int(xi)) + 1 if xi != 0 else 0 for xi in self.intensity
        )
        self.intensity = np.fromiter(iterable, float)

    def show_fitness(self, vff) -> None:
        vff_str = ""
        for firefly in vff:
            vff_str = f"{vff_str} {firefly.fitness:.2f} "
        self.my_print(vff_str)
        # print(vff_str)

    def my_print(self, string):
        if self.show:
            print(string)

    def __str__(self):
        result = f"FireFly_HFA:-pop_size:{self.pop_size}, crossover_rate:{self.crossover_rate}, mutation_rate:{self.mutation_rate}"
        return result
