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

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import CVRProblem
from algorithms import FireFlyHFA
from pathlib import Path
import time
import matplotlib
from concurrent.futures import ProcessPoolExecutor

# matplotlib.use('Agg')    # Computador Personal
# matplotlib.use('QtAgg')
# matplotlib.use('TkAgg')    # Portatil


# Plot convergence curve
def plot_convergence_curve(fitness_history, p, alg):
    efos = np.arange(len(fitness_history))
    plt.title("Convergence curve for " + str(p))
    plt.xlabel("Iteractions")
    plt.ylabel("Fitness")
    plt.plot(efos, fitness_history, label=str(alg))
    plt.legend()
    plt.show()


def plot_convergence_curve_comparison(dir_result, fitness_history, start_iter, f, alg):
    iter = np.arange(start_iter, len(fitness_history[0]) + start_iter)
    plt.title(f"Convergence curve for {f}")
    plt.xlabel("Iteractions")
    plt.ylabel("Fitness")
    algorithms = len(fitness_history)
    for a in range(algorithms):
        plt.plot(iter, fitness_history[a], label=str(alg[a]))
    plt.legend()
    # plt.show()
    plt.savefig(f"{dir_result}Comparison_{f}_{alg}.png")
    plt.clf()
    plt.close()


def plot_all_curves(namefile, all_curves, start_iter, f):
    iter = np.arange(start_iter, len(all_curves[0]))
    plt.figure(figsize=(15, 9))
    plt.title(f"curve for {f}")
    plt.xlabel("Iteractions")
    plt.ylabel("Fitness")
    curves = len(all_curves)
    for c in range(curves - 1):
        plt.plot(iter, all_curves[c, start_iter:], "--", label=str(c))
    plt.plot(iter, all_curves[curves - 1, start_iter:], "-", label="m")
    plt.legend()
    # plt.show()
    plt.savefig(f"{namefile}.png")
    plt.clf()
    plt.close()


def print_algorithms_with_avg_fitness(alg, avg_fitness):
    rows = len(alg)
    for r in range(rows):
        print(alg[r] + " {0:12.6f}".format(avg_fitness[r]))


def main_FireFly_HFA(
    problems,
    repetitions=10,
    max_iter=20,
    pop_size=110,
    crossover_rate=0.95,
    mutation_rate=0.1,
    ff_comp=0,
    dir_result_base="./results/",
):

    list_ff_comp = ["BSF", "PrevBest", "PairBE", "Rand"]
    start_timer_all = time.time()

    namefile = f"FireFly{list_ff_comp[ff_comp]}_rep{repetitions}_iter{max_iter}_pob{pop_size}_cr{int(crossover_rate*100)}_mr{int(mutation_rate*100)}"
    dir_result = f"{dir_result_base}/{namefile}/"
    print(dir_result)
    path = Path(dir_result)
    path.mkdir(parents=True, exist_ok=True)
    dir_result_sol = f"{dir_result}/solution/"
    path_sol = Path(dir_result_sol)
    path_sol.mkdir(parents=True, exist_ok=True)

    print(
        f"\nEjemplo Principal.\n"
        f"rep{repetitions}_iter{max_iter}_pob{pop_size}_cr{int(crossover_rate*100)}_mr{int(mutation_rate*100)}"
    )

    firefly_hfa = FireFlyHFA(
        max_iter=max_iter,
        pop_size=pop_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        ff_comp=ff_comp,
    )

    algorithms = [firefly_hfa]

    df = pd.DataFrame(
        {
            "Problem": pd.Series(dtype="str"),
            "BKS": pd.Series(dtype="int"),
            "Best HFA": pd.Series(dtype="float"),
            "BGap HFA": pd.Series(dtype="float"),
            "Avg. HFA": pd.Series(dtype="float"),
            "AGap HFA": pd.Series(dtype="float"),
            "Iter HFA": pd.Series(dtype="int"),
        }
    )
    df2 = pd.DataFrame(
        {"Problem": pd.Series(dtype="str"), "HFA": pd.Series(dtype="float")}
    )
    df_time = pd.DataFrame(
        {"Problem": pd.Series(dtype="str"), "HFA": pd.Series(dtype="float")}
    )
    str_best_solution = ""
    vec_repetition = np.arange(repetitions)

    vec_seed = []
    vec_problems = []
    vec_alg = []
    for alg in algorithms:
        alg.show = False
    for p in problems:
        for alg in algorithms:
            vec_alg.append(alg)
            for i in vec_repetition:
                vec_seed.append(i)
                vec_problems.append(p)
    for i in range(len(vec_seed)):
        print(f"{i}. {vec_seed[i]} -->  {vec_problems[i]}")

    # executor = ProcessPoolExecutor(max_workers=12)
    executor = ProcessPoolExecutor(10)
    alg = vec_alg[0]
    # futures = list(executor.map(alg.evolve, vec_seed, vec_problems))
    name_algorithm = "HFA"
    new_futures = []

    for result in executor.map(alg.solve, vec_seed, vec_problems):
        curve_data, best_firelfy, b_iter, time_spend, name_prob, seed = result
        # print(f"\n\n {name_prob}, {seed} {time_spend}\n")
        best_firelfy.save(
            f"{dir_result_sol}{name_prob}_{name_algorithm}_seed_{seed:02d}.sol"
        )
        new_futures.append((curve_data, best_firelfy, b_iter, time_spend))
    # sys.exit("Fin")
    index = 0
    for p in problems:
        bks = p.bestKnown
        names_alg = []
        avg_curve_alg = []
        best_solution_alg = []
        best_solution_alg_bgap = []
        best_avg_fitness_alg = []
        best_avg_fitness_alg_agap = []
        best_solution_alg_iter = []
        best_std_fitness_alg = []
        alg_time = []
        print("*" * 60)
        print(f"Problema: {p}")
        start_iter = 1

        for alg in algorithms:

            avg_curve = np.zeros(max_iter + 1, float)
            all_curves = np.zeros((repetitions + 1, max_iter + 1), float)
            best_fitnes = np.zeros(repetitions, float)
            best_iter = np.zeros(repetitions, int)

            time_by_repetition = 0
            name_algorithm = str(alg).split(":")[0]

            for s in range(0, repetitions):
                curve_data, best_firelfy, b_iter, time_spend = new_futures[index]
                index += 1
                all_curves[s] = curve_data
                time_by_repetition = time_by_repetition + time_spend
                avg_curve = avg_curve + curve_data
                best_fitnes[s] = curve_data[-1]
                best_iter[s] = b_iter

            avg_curve = avg_curve / repetitions
            all_curves[repetitions] = avg_curve
            df_all_curves = pd.DataFrame(all_curves)
            print(df_all_curves.to_string())
            namefile_all_curves = f"{dir_result}All_curves_{p}_{name_algorithm}"
            df_all_curves.to_csv(f"{namefile_all_curves}.csv")

            plot_all_curves(f"{namefile_all_curves}", all_curves, start_iter, p)

            avg_best_fitnes = np.average(best_fitnes)
            std_best_fitnes = np.std(best_fitnes)
            ind_best_solution = np.argmin(best_fitnes)
            best_solution = best_fitnes[ind_best_solution]
            best_experiments_mask = np.isclose(best_fitnes, best_solution)
            best_solution_iter = np.min(best_iter[best_experiments_mask])
            ind_best_solution = np.argmin(best_fitnes)
            str_best_solution += f"{p.file_name}_{name_algorithm} {best_solution} index: {ind_best_solution}\n"

            names_alg.append(name_algorithm)

            avg_curve_alg.append(avg_curve[start_iter:])
            best_solution_alg.append(int(best_solution))
            best_solution_alg_bgap.append((int(best_solution) - bks) / bks * 100)
            best_avg_fitness_alg.append(avg_best_fitnes)
            best_avg_fitness_alg_agap.append((avg_best_fitnes - bks) / bks * 100)
            best_solution_alg_iter.append(best_solution_iter)
            best_std_fitness_alg.append(std_best_fitnes)
            alg_time.append(time_by_repetition)
            print(f"\nProblema: {p}")
            print(f"Algoritmo {alg}, Tiempo de ejecución: {time_by_repetition:.4f}")
            print(f"Best: {best_solution}, avg_best: {avg_best_fitnes}\n")

        plot_convergence_curve_comparison(
            f"{dir_result}/", avg_curve_alg, start_iter, p, names_alg
        )

        new_row = pd.DataFrame(
            {
                "Problem": str(p.file_name),
                "BKS": f"{p.bestKnown}",
                "Best HFA": f"{best_solution_alg[0]}",
                "BGap HFA": f"{best_solution_alg_bgap[0]:.2f}",
                "Avg. HFA": f"{best_avg_fitness_alg[0]:.1f}",
                "AGap HFA": f"{best_avg_fitness_alg_agap[0]:.2f}",
                "Iter HFA": f"{best_solution_alg_iter[0]}",
            },
            index=[0],
        )
        df = pd.concat([df.loc[:], new_row]).reset_index(drop=True)

        new_row2 = pd.DataFrame(
            {"Problem": str(p.file_name), "HFA": f"{best_std_fitness_alg[0]:.1f}"},
            index=[0],
        )
        df2 = pd.concat([df2.loc[:], new_row2]).reset_index(drop=True)

        new_row_time = pd.DataFrame(
            {"Problem": str(p.file_name), "HFA": f"{alg_time[0]:.2f}"}, index=[0]
        )
        df_time = pd.concat([df_time.loc[:], new_row_time]).reset_index(drop=True)

    df_str = df.to_string()
    print(
        df_str,
        f"\nSave in: {dir_result}/avgs_{namefile}.txt'\nTime: {time.time() - start_timer_all}\n",
    )
    f = open(f"{dir_result}/avgs_{namefile}.txt", "a")
    f.write(f"\n{dir_result}\nTime: {time.time() - start_timer_all}\n")
    f.write(df_str)
    f.close()

    f = open(f"{dir_result_base}/avgs_all.txt", "a")
    f.write(f"\n{dir_result}\nTime{time.time() - start_timer_all}\n")
    f.write(df_str)
    f.close()

    df.to_csv(f"{dir_result}/avgs_{namefile}.csv", index=False)

    print("Desviaciones Estandar:")
    print(df2.to_string())
    df2.to_csv(f"{dir_result}/stds_{namefile}.csv", index=False)
    print("Tiempo empleado")
    print(df_time.to_string())
    df_time.to_csv(f"{dir_result}/time_{namefile}.csv", index=False)
    print(str_best_solution)
    f = open(f"{dir_result}/best_solutions.txt", "a")
    f.write(str_best_solution)
    f.close()
