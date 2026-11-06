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
import traceback
import numpy as np
import itertools
from utils import CVRProblem
import vrplib


class VehicleRoute:
    def __init__(self):
        self.route = [0]
        self.fitness = 0.0
        self.capacity = 0
        self.sx = 0
        self.sy = 0
        self.theta = 0
        self.r = 0

    def __lt__(self, other):
        # return self.fitness > other.fitness  # descendent
        return self.theta < other.theta  # ascendente

    def __str__(self):
        return f"Route:{self.route} fitness:{self.fitness} capacity:{self.capacity}"


class SolutionHFA:
    def __init__(self, p: CVRProblem, sort_theta: bool = True):
        self.problem = p
        self.sort_theta = sort_theta
        self.routes = np.empty(self.problem.vehicles, dtype=VehicleRoute)
        for i in range(self.problem.vehicles):
            self.routes[i] = VehicleRoute()
        self.fitness = 0.0
        self.tsp_sr = None

    def __lt__(self, other):
        return self.fitness < other.fitness

    def from_solution(self, origin):
        self.problem = origin.problem
        for i in range(origin.routes.size):
            self.routes[i].route = origin.routes[i].route.copy()
            self.routes[i].fitness = origin.routes[i].fitness
            self.routes[i].capacity = origin.routes[i].capacity
            self.routes[i].sx = origin.routes[i].sx
            self.routes[i].sy = origin.routes[i].sy
            self.routes[i].r = origin.routes[i].r
            self.routes[i].theta = origin.routes[i].theta
        self.tsp_sr = origin.tsp_sr.copy()
        self.fitness = origin.fitness

    def initialization(self):
        correct = False
        iter = 0
        while (not correct) and iter < 100000:
            iter += 1
            size = self.problem.size
            customers = np.random.choice(
                range(1, size), self.problem.size - 1, replace=False
            )
            correct = self.initialization_from(customers)
            if not correct:
                for i in range(self.problem.vehicles):
                    self.routes[i].route = [0]
                    self.routes[i].capacity = 0
                    self.routes[i].fitness = 0
                    self.routes[i].sx = 0
                    self.routes[i].sy = 0
                    self.routes[i].theta = 0
                    self.routes[i].r = 0
                self.fitness = 0

        if correct:
            for i in range(self.problem.vehicles):
                self.routes[i].route = np.array(
                    self.routes[i].route, dtype=VehicleRoute
                )

            self.fitness = sum(
                self.routes[i].fitness for i in range(self.problem.vehicles)
            )
            self.tsp_sr = self.to_tsp_sr()
            self.routes = np.sort(self.routes)
            self.check()
        else:
            print(
                f"Problema grave en {self.problem.file_name_c}, no se pudo generar en {iter} intentos "
            )
            input()

    def initialization_from(self, global_route):
        n_route = 0
        for customerX in global_route:
            customer = int(customerX)
            if (
                self.routes[n_route].capacity + self.problem.demand[customer]
                > self.problem.capacity
            ):
                if n_route + 1 >= self.problem.vehicles:
                    self.fitness = float("inf")
                    return False
                last = self.routes[n_route].route[-1]
                self.routes[n_route].route.append(0)
                self.routes[n_route].fitness += self.problem.distances[last, 0]
                n_route += 1
            last = self.routes[n_route].route[-1]
            self.routes[n_route].route.append(customer)
            self.routes[n_route].fitness += self.problem.distances[last, customer]
            self.routes[n_route].capacity += self.problem.demand[customer]
        last = self.routes[n_route].route[-1]
        self.routes[n_route].route.append(0)
        self.routes[n_route].fitness += self.problem.distances[last, 0]
        for i in range(self.problem.vehicles):
            self.routes[i].route = np.array(self.routes[i].route, dtype=int)
        self.fitness = sum(self.routes[i].fitness for i in range(self.problem.vehicles))

        self.tsp_sr = self.to_tsp_sr()
        return True

    def to_tsp_sr(self):
        vec_tsp = self.routes[0].route[1:-1]
        for r in self.routes[1:]:
            vec_tsp = np.concatenate([vec_tsp, r.route[1:-1]])
        return vec_tsp

    def hd(self, other):
        # hamming distance
        h_distance = 0
        if len(self.tsp_sr) == len(other.tsp_sr):
            h_distance += np.sum(self.tsp_sr != other.tsp_sr)
        else:
            print(f"\nError HD, len: {len(self.tsp_sr)}\n{len(other.tsp_sr)}")
            input()
        return h_distance

    def save(self, name_file):
        routes = []
        for i in range(self.problem.vehicles):
            routes.append(list(self.routes[i].route[1:-1]))
        solution_data = {"Cost": f"{self.evaluate_round()}"}
        vrplib.write_solution(name_file, routes, solution_data)

    def sort_routes(self):
        if self.sort_theta:
            for i in range(self.problem.vehicles):
                if self.routes[i].route.size > 3:  # OJO revisar despues
                    first = self.problem.sort_cust_polar[self.routes[i].route[1]]
                    last = self.problem.sort_cust_polar[self.routes[i].route[-2]]
                    if first > last:
                        self.routes[i].route[:] = self.routes[i].route[:][::-1]
                self.routes[i].theta = self.problem.theta[self.routes[i].route[1]]
            self.routes = np.sort(self.routes)
        else:
            self.routes = np.sort(self.routes)
            for i in range(self.problem.vehicles):
                if self.routes[i].route.size > 3:  # OJO revisar despues
                    if self.routes[i].route[1] > self.routes[i].route[-2]:
                        self.routes[i].route[:] = self.routes[i].route[:][::-1]

    def partially_matched_crossover(
        self, parent1, parent2, first_point=None, second_point=None
    ):
        first_point = np.random.randint(0, len(parent1) - 1)
        second_point = np.random.randint(first_point + 1, len(parent1))

        left_p1 = parent1[:first_point]
        middle_p1 = parent1[first_point:second_point]
        right_p1 = parent1[second_point:]

        left_p2 = parent2[:first_point]
        middle_p2 = parent2[first_point:second_point]
        right_p2 = parent2[second_point:]

        relation1 = self.relation(middle_p1, middle_p2)
        relation2 = self.relation(middle_p2, middle_p1)

        middle_p1_set = set(middle_p1)
        middle_p2_set = set(middle_p2)

        left_p1_rel2 = np.array(
            [relation2[x] if x in middle_p2_set else x for x in left_p1]
        )
        right_p1_rel2 = np.array(
            [relation2[x] if x in middle_p2_set else x for x in right_p1]
        )

        left_p2_rel1 = np.array(
            [relation1[x] if x in middle_p1_set else x for x in left_p2]
        )
        right_p2_rel1 = np.array(
            [relation1[x] if x in middle_p1_set else x for x in right_p2]
        )

        aux_h1 = np.concatenate([left_p1_rel2, middle_p2, right_p1_rel2])
        aux_h2 = np.concatenate([left_p2_rel1, middle_p1, right_p2_rel1])
        return aux_h1, aux_h2

    def relation(self, middle_p1, middle_p2):
        dic_tem1 = {}
        for i in range(len(middle_p1)):
            dic_tem1[middle_p1[i]] = middle_p2[i]

        dict_rel = {}
        for e in dic_tem1:
            dict_rel[e] = dic_tem1[e]
            aux = dict_rel[e]
            while aux in dic_tem1:
                y = dic_tem1[aux]
                if y == e:
                    break
                else:
                    dict_rel[e] = y
                aux = dict_rel[e]

        return dict_rel

    def calc_distance(self, solution, tsp_sr):
        distances = self.problem.distances

        if len(solution) <= 1:
            distance = 0.0
        else:
            distance = distances[solution[:-1], solution[1:]].sum()

        if tsp_sr:
            distance += distances[0, solution[0]] + distances[solution[-1], 0]

        return distance

    def check(self):
        return
        if len(self.tsp_sr) + 1 != self.problem.size:
            print(self)
            traceback.print_stack()
            sys.exit(f"\nCheck error, size {len(self.tsp_sr) + 1}")

        for i in range(1, self.problem.size):
            if not i in self.tsp_sr:
                print(self)
                traceback.print_stack()
                sys.exit(f"\nCheck error, {i} not in {self.tsp_sr}")
        fitness = 0.0
        capacity = 0
        for i in range(self.problem.vehicles):
            fit = 0.0
            cap = 0
            last_customer = 0
            list_dis = []
            for customer in self.routes[i].route[1:]:
                cap += self.problem.demand[customer]
                aux = self.problem.distances[last_customer, customer]
                fit += aux
                list_dis.append(f"{aux:.2f}")
                last_customer = customer
            fitness += fit
            capacity += cap
            if (cap > self.problem.capacity) or (cap != self.routes[i].capacity):
                print(i, self.routes[i].route)
                print(
                    f"Capacity calculate: {cap},  {[self.problem.demand[customer] for customer in self.routes[i].route]}"
                )
                print(self)
                traceback.print_stack()
                sys.exit(
                    f"\nCapacity error, route: {i}, capacity cal: {cap}, actual: {self.routes[i].capacity}."
                )

            if abs(fit - self.routes[i].fitness) > 1e-4:
                print(i, self.routes[i].route)
                print(f"Fitness: {fit}, {list_dis}")
                print(self)
                traceback.print_stack()
                sys.exit(
                    f"\nFitness error, route: {i}, fitness cal: {fit}, actual: {self.routes[i].fitness}."
                )

        if abs(fitness - self.fitness) > 1e-5:
            print(self)
            traceback.print_stack()
            sys.exit(f"\nFitness total error., Cal: {fitness}, actual: {self.fitness}")

    def evaluate(self):
        fitness = 0
        distances = self.problem.distances
        routes = self.routes
        vehicles = self.problem.vehicles

        for i in range(vehicles):
            route = routes[i].route
            for j in range(1, len(route)):
                c1 = route[j - 1]
                c2 = route[j]
                fitness += distances[c1, c2]

        return fitness

    def evaluate_round(self):
        fitness = 0
        for i in range(self.problem.vehicles):
            for j in range(1, len(self.routes[i].route)):
                c1 = self.routes[i].route[j - 1]
                c2 = self.routes[i].route[j]
                fitness += round(self.problem.distances[c1, c2])
        return fitness

    def __str__(self):
        str_sol = f"\n{self.problem}"
        for i in range(self.problem.vehicles):
            str_sol = f"{str_sol}\n{i}: {self.routes[i]}"
        str_sol = f"{str_sol}\nfitness: {self.fitness}"
        return str_sol

    def initialization_Sweep1(self):
        if self.problem.sort_cust_polar[0] == 0:
            attempts = 0
            correct = False
            customers = self.problem.sort_cust_polar[1:]

            rand_customers = np.random.choice(
                np.arange(customers.size), size=customers.size, replace=False
            )

            while attempts <= self.problem.size and not correct:
                start_index = rand_customers[attempts]
                actual_customers = np.concatenate(
                    [customers[start_index:], customers[:start_index]]
                )
                attempts += 1
                correct = self.initialization_several_Sweep_fromX(actual_customers, 10)
                if not correct:
                    print(
                        f"  {'<NOT> ' * 10},  attempts: {attempts}, start_index: {start_index}, {self.problem.file_name_c}"
                    )

            if not correct:
                print(
                    f"\nFail initialization_several_Sweep_fromX, start_index: {start_index}"
                )
                self.initialization()
        else:
            print(f"\nError, Inicilization name Problem: {self.problem.file_name_c}")
            print(self.problem.sort_cust_polar)
            self.initialization()

    def initialization_several_Sweep_fromX(self, global_route, maximum_attempts=10):
        attempts = 0
        found_successful_route = False
        actual_global_route = np.ones(global_route.size, dtype=int)
        while (attempts < maximum_attempts) and not found_successful_route:
            for i in range(self.problem.vehicles):
                self.routes[i].route = [0]
                self.routes[i].capacity = 0
                self.routes[i].fitness = 0
            self.fitness = 0
            actual_global_route[:] = -1
            actual_global_route[0] = global_route[0]
            changed = False
            for i in range(global_route.size - 1):
                if changed:
                    changed = False
                else:
                    rand_int = np.random.randint(2)
                    if rand_int == 0:
                        # Copy
                        actual_global_route[i], actual_global_route[i + 1] = (
                            global_route[i],
                            global_route[i + 1],
                        )
                        changed = False
                    else:
                        # Interchange
                        actual_global_route[i], actual_global_route[i + 1] = (
                            global_route[i + 1],
                            global_route[i],
                        )
                        changed = True
            if actual_global_route[-1] == -1:
                actual_global_route[-1] = global_route[-1]
            attempts += 1

            found_successful_route = self.initialization_with_repair_from(
                actual_global_route
            )
        # self.check()
        return found_successful_route

    def opt_pair(self):
        dist = self.problem.distances
        for index_route in range(self.problem.vehicles):
            cont = 1
            while cont > 0:
                route = self.routes[index_route].route.copy()  # [1:-1]
                best_solution = route.copy()
                best_distance = self.routes[index_route].fitness
                cont = 0
                for i in range(1, route.size - 1):
                    j = i + 1
                    new_solution = np.concatenate(
                        [route[:i], route[i : j + 1][::-1], route[j + 1 :]]
                    )
                    new_distance = self.calc_distance(new_solution, False)
                    if new_distance < best_distance:
                        best_solution = new_solution
                        best_distance = new_distance
                        cont += 1
                self.routes[index_route].route = best_solution
                self.routes[index_route].fitness = best_distance
        self.fitness = sum(self.routes[i].fitness for i in range(self.problem.vehicles))
        self.tsp_sr = self.to_tsp_sr()
        self.check()

    def change_one(self, repetition=2):
        dist = self.problem.distances
        for _ in range(repetition):
            for index_route in range(self.problem.vehicles):
                route = self.routes[index_route].route.copy()  # [1:-1]

                best_solution = route.copy()
                best_distance = self.routes[index_route].fitness
                prev_dist = best_distance
                for k in np.random.choice(
                    range(1, route.size - 1), size=route.size - 2, replace=False
                ):
                    j = np.random.randint(low=1, high=route.size - 1)
                    i = k
                    if i > j:
                        i, j = j, i
                    if i != j:
                        ele_i = route[i]
                        cost_actual = (
                            dist[route[i - 1], ele_i]
                            + dist[ele_i, route[i + 1]]
                            + dist[route[j], route[j + 1]]
                        )
                        cost_new = (
                            dist[route[i - 1], route[i + 1]]
                            + dist[route[j], ele_i]
                            + dist[ele_i, route[j + 1]]
                        )
                        new_distance = prev_dist + cost_new - cost_actual
                        if (
                            new_distance < best_distance
                            and abs(new_distance - best_distance) > 1e-8
                        ):
                            new_solution = np.concatenate(
                                [
                                    route[:i],
                                    route[i + 1 : j + 1],
                                    [ele_i],
                                    route[j + 1 :],
                                ]
                            )
                            best_solution = new_solution
                            best_distance = new_distance
                self.routes[index_route].route = best_solution
                self.routes[index_route].fitness = best_distance

        self.fitness = sum(self.routes[i].fitness for i in range(self.problem.vehicles))
        self.tsp_sr = self.to_tsp_sr()
        self.check()
        self.change_two()
        self.one_elem_two_routes()
        self.two_elem_two_routes()

    def new_2_opt(self, repetition=2):
        dist = self.problem.distances
        routes = self.routes
        vehicles = self.problem.vehicles

        # print(self)
        self.check()

        improved = False

        for _ in range(repetition):
            for index_route in range(vehicles):
                route = routes[index_route].route.copy()  # [1:-1]

                best_solution = route
                best_distance = routes[index_route].fitness
                prev_dist = best_distance

                for k in np.random.choice(
                    range(1, route.size - 1), size=route.size - 2, replace=False
                ):
                    # if np.random.randint(5)!=0:
                    #     continue

                    j = np.random.randint(low=1, high=route.size - 1)

                    i = k
                    if i > j:
                        i, j = j, i

                    if i != j:
                        cost_actual = (
                            dist[route[i - 1], route[i]] + dist[route[j], route[j + 1]]
                        )
                        cost_new = (
                            dist[route[i - 1], route[j]] + dist[route[i], route[j + 1]]
                        )

                        new_distance = prev_dist + cost_new - cost_actual

                        if (new_distance < best_distance) and abs(
                            new_distance - best_distance
                        ) > 1e-8:
                            new_solution = np.concatenate(
                                [route[:i], route[i : j + 1][::-1], route[j + 1 :]]
                            )
                            best_solution = new_solution
                            best_distance = new_distance
                            improved = True
                            # route = new_solution

                routes[index_route].route = best_solution
                routes[index_route].fitness = best_distance

        if improved:
            self.fitness = sum(routes[i].fitness for i in range(vehicles))
            self.tsp_sr = self.to_tsp_sr()
        self.check()

    def change_two(self, repetition=2):
        dist = self.problem.distances

        for _ in range(repetition):
            for index_route in range(self.problem.vehicles):
                route = self.routes[index_route].route.copy()

                best_solution = route.copy()
                best_distance = self.routes[index_route].fitness
                base_distance = best_distance

                for k in np.random.choice(
                    range(1, route.size - 1), size=route.size - 2, replace=False
                ):
                    j = np.random.randint(low=1, high=route.size - 1)

                    i = k
                    if i > j:
                        i, j = j, i

                    if i != j:
                        c1 = route[i]
                        c2 = route[j]

                        c1prev = route[i - 1]
                        c1next = route[i + 1]
                        c2prev = route[j - 1]
                        c2next = route[j + 1]

                        if j == i + 1:
                            old_dist = (
                                dist[c1prev, c1] + dist[c1, c2] + dist[c2, c2next]
                            )
                            new_dist = (
                                dist[c1prev, c2] + dist[c2, c1] + dist[c1, c2next]
                            )
                        else:
                            old_dist = (
                                dist[c1prev, c1]
                                + dist[c1, c1next]
                                + dist[c2prev, c2]
                                + dist[c2, c2next]
                            )
                            new_dist = (
                                dist[c1prev, c2]
                                + dist[c2, c1next]
                                + dist[c2prev, c1]
                                + dist[c1, c2next]
                            )

                        new_distance = base_distance + new_dist - old_dist

                        if (
                            new_distance < best_distance
                            and abs(new_distance - best_distance) > 1e-8
                        ):
                            new_solution = route.copy()
                            new_solution[i], new_solution[j] = (
                                new_solution[j],
                                new_solution[i],
                            )
                            best_solution = new_solution
                            best_distance = new_distance

                self.routes[index_route].route = best_solution
                self.routes[index_route].fitness = best_distance

        self.fitness = sum(self.routes[i].fitness for i in range(self.problem.vehicles))
        self.tsp_sr = self.to_tsp_sr()
        self.check()

    def one_elem_two_routes(self, repetitions=2):
        routes = self.routes
        problem = self.problem
        vehicles = problem.vehicles
        capacity_limit = problem.capacity
        demand = problem.demand
        dist = problem.distances

        route_indices = np.arange(0, vehicles)
        route_choices = [route_indices[route_indices != r] for r in range(vehicles)]

        improved = False

        for index_route in range(vehicles):
            n_route1 = index_route

            for _ in range(repetitions):
                n_route2 = np.random.choice(route_choices[n_route1])

                len_route1 = len(routes[n_route1].route) - 2
                len_route2 = len(routes[n_route2].route) - 2

                if (len_route1 <= 0) or (len_route2 <= 0):
                    print(n_route1, routes[n_route1].route)
                    print(n_route2, routes[n_route2].route)
                    print(self)
                    input("Cazando errorres in one_elem_two_routes")
                    return

                index1 = np.random.randint(len_route1) + 1
                c1 = routes[n_route1].route[index1]
                d1 = demand[c1]

                index2 = np.random.randint(len_route2) + 1
                c2 = routes[n_route2].route[index2]

                cap_r1 = routes[n_route1].capacity - d1
                cap_r2 = routes[n_route2].capacity + d1

                if cap_r2 < capacity_limit:
                    route1 = routes[n_route1].route
                    route2 = routes[n_route2].route

                    c1prev = route1[index1 - 1]
                    c1next = route1[index1 + 1]

                    c2prev = route2[index2 - 1]

                    old_dist1 = dist[c1prev, c1] + dist[c1, c1next]
                    new_dist1 = dist[c1prev, c1next]

                    old_dist2 = dist[c2prev, c2]
                    new_dist2 = dist[c2prev, c1] + dist[c1, c2]

                    if (new_dist1 + new_dist2) < (old_dist1 + old_dist2):
                        routes[n_route1].route = np.concatenate(
                            [route1[:index1], route1[index1 + 1 :]]
                        )
                        routes[n_route1].capacity = cap_r1
                        routes[n_route1].fitness += new_dist1 - old_dist1

                        routes[n_route2].route = np.concatenate(
                            [route2[:index2], [c1], route2[index2:]]
                        )
                        routes[n_route2].capacity = cap_r2
                        routes[n_route2].fitness += new_dist2 - old_dist2

                        improved = True

        if improved:
            self.fitness = sum(routes[i].fitness for i in range(vehicles))
            self.tsp_sr = self.to_tsp_sr()
            self.check()

    def two_elem_two_routes(self, repetitions=2):
        routes = self.routes
        problem = self.problem
        vehicles = problem.vehicles
        capacity_limit = problem.capacity
        demand = problem.demand
        dist = problem.distances

        route_indices = np.arange(0, vehicles)
        route_choices = [route_indices[route_indices != r] for r in range(vehicles)]

        improved = False

        for index_route in range(vehicles):
            n_route1 = index_route

            for _ in range(repetitions):
                n_route2 = np.random.choice(route_choices[n_route1])

                len_route1 = len(routes[n_route1].route) - 2
                len_route2 = len(routes[n_route2].route) - 2

                if (len_route1 <= 0) or (len_route2 <= 0):
                    print(n_route1, routes[n_route1].route)
                    print(n_route2, routes[n_route2].route)
                    print(self)
                    input("Cazando errorres in two_elem_two_routes")
                    return

                index1 = np.random.randint(len_route1) + 1
                c1 = routes[n_route1].route[index1]
                d1 = demand[c1]

                index2 = np.random.randint(len_route2) + 1
                c2 = routes[n_route2].route[index2]
                d2 = demand[c2]

                cap_r1 = routes[n_route1].capacity - d1 + d2
                cap_r2 = routes[n_route2].capacity - d2 + d1

                if (cap_r1 < capacity_limit) and (cap_r2 < capacity_limit):
                    route1 = routes[n_route1].route
                    route2 = routes[n_route2].route

                    c1prev = route1[index1 - 1]
                    c1next = route1[index1 + 1]

                    c2prev = route2[index2 - 1]
                    c2next = route2[index2 + 1]

                    old_dist1 = dist[c1prev, c1] + dist[c1, c1next]
                    new_dist1 = dist[c1prev, c2] + dist[c2, c1next]

                    old_dist2 = dist[c2prev, c2] + dist[c2, c2next]
                    new_dist2 = dist[c2prev, c1] + dist[c1, c2next]

                    if (new_dist1 + new_dist2) < (old_dist1 + old_dist2):
                        route1[index1] = c2
                        routes[n_route1].capacity = cap_r1
                        routes[n_route1].fitness += new_dist1 - old_dist1

                        route2[index2] = c1
                        routes[n_route2].capacity = cap_r2
                        routes[n_route2].fitness += new_dist2 - old_dist2

                        improved = True

        if improved:
            self.fitness = sum(routes[i].fitness for i in range(vehicles))
            self.tsp_sr = self.to_tsp_sr()
            self.check()

    def local_searchHFA(self):
        x_new = Solution(self.problem)
        for i in range(self.problem.vehicles):
            x_new.routes[i].route, x_new.routes[i].fitness = self.tsp_improved_2_opt(
                self.routes[i]
            )
            x_new.routes[i].capacity = self.routes[i].capacity
        x_new.fitness = sum(
            x_new.routes[i].fitness for i in range(x_new.problem.vehicles)
        )
        x_new.tsp_sr = x_new.to_tsp_sr()
        x_new.check()
        self.from_solution(x_new)

    def mutationHFA(self, mutation_rate):
        sol_ok = False
        x_new = Solution(self.problem)
        if np.random.random() < mutation_rate:
            self.check()
            y = self.tsp_2_hand_opt(self.tsp_sr)
            sol_ok = x_new.initialization_from(y)
        if sol_ok:
            x_new.check()
            self.from_solution(x_new)

        if np.random.random() <= mutation_rate:
            self.check()
            if np.random.random() <= 0.5:
                n_route = np.random.randint(self.problem.vehicles)
                len_route = len(self.routes[n_route].route) - 2
                index1 = np.random.randint(len_route) + 1
                index2 = np.random.randint(len_route) + 1
                if index1 > index2:
                    index1, index2 = index2, index1
                if index1 != index2:
                    c1 = self.routes[n_route].route[index1]
                    c1prev = self.routes[n_route].route[index1 - 1]
                    c1next = self.routes[n_route].route[index1 + 1]
                    c2 = self.routes[n_route].route[index2]
                    c2prev = self.routes[n_route].route[index2 - 1]
                    c2next = self.routes[n_route].route[index2 + 1]
                    dist = self.problem.distances
                    if abs(index1 - index2) == 1:
                        old_dist = dist[c1prev, c1] + dist[c2, c2next]
                        new_dist = dist[c1prev, c2] + dist[c1, c2next]
                    else:
                        old_dist = (
                            dist[c1prev, c1]
                            + dist[c1, c1next]
                            + dist[c2prev, c2]
                            + dist[c2, c2next]
                        )
                        new_dist = (
                            dist[c1prev, c2]
                            + dist[c2, c1next]
                            + dist[c2prev, c1]
                            + dist[c1, c2next]
                        )

                    self.routes[n_route].route[index1] = c2
                    self.routes[n_route].route[index2] = c1
                    self.routes[n_route].fitness += new_dist - old_dist

                self.fitness = sum(
                    self.routes[i].fitness for i in range(self.problem.vehicles)
                )
                self.tsp_sr = self.to_tsp_sr()
                self.check()
            else:
                n_route1, n_route2 = np.random.choice(
                    np.arange(0, self.problem.vehicles), size=2, replace=False
                )
                len_route1 = len(self.routes[n_route1].route) - 2
                len_route2 = len(self.routes[n_route2].route) - 2
                if (len_route1 <= 0) or (len_route2 <= 0):
                    print(n_route1, self.routes[n_route1].route)
                    print(n_route2, self.routes[n_route2].route)
                    print(self)
                    input("Cazando errorres...")
                    return

                index1 = np.random.randint(len_route1) + 1
                c1 = self.routes[n_route1].route[index1]
                d1 = self.problem.demand[c1]

                index2 = np.random.randint(len_route2) + 1
                c2 = self.routes[n_route2].route[index2]
                d2 = self.problem.demand[c2]
                cap_r1 = self.routes[n_route1].capacity - d1 + d2
                cap_r2 = self.routes[n_route2].capacity - d2 + d1
                if (cap_r1 < self.problem.capacity) and (
                    cap_r2 < self.problem.capacity
                ):
                    # c1 = self.routes[n_route1].route[index1]
                    c1prev = self.routes[n_route1].route[index1 - 1]
                    c1next = self.routes[n_route1].route[index1 + 1]
                    # c2 = self.routes[n_route2].route[index2]
                    c2prev = self.routes[n_route2].route[index2 - 1]
                    c2next = self.routes[n_route2].route[index2 + 1]
                    dist = self.problem.distances
                    old_dist1 = dist[c1prev, c1] + dist[c1, c1next]
                    new_dist1 = dist[c1prev, c2] + dist[c2, c1next]
                    old_dist2 = dist[c2prev, c2] + dist[c2, c2next]
                    new_dist2 = dist[c2prev, c1] + dist[c1, c2next]

                    self.routes[n_route1].route[index1] = c2
                    self.routes[n_route1].capacity = cap_r1
                    self.routes[n_route1].fitness += new_dist1 - old_dist1
                    self.routes[n_route2].route[index2] = c1
                    self.routes[n_route2].capacity = cap_r2
                    self.routes[n_route2].fitness += new_dist2 - old_dist2

                    # self.fitness = self.evaluate() # 2026
                    self.fitness = sum(
                        self.routes[i].fitness for i in range(self.problem.vehicles)
                    )
                    self.tsp_sr = self.to_tsp_sr()
                    self.check()

    def init_local(self, repetition=1):
        dist = self.problem.distances
        # print(self)
        self.check()
        for _ in range(repetition):
            total_fit = 0.0
            for index_route in range(self.problem.vehicles):
                self.routes[index_route].route[1:-1] = np.random.choice(
                    self.routes[index_route].route[1:-1],
                    size=self.routes[index_route].route.size - 2,
                    replace=False,
                )
                fit = 0.0
                last_customer = 0
                for customer in self.routes[index_route].route[1:]:
                    aux = dist[last_customer, customer]
                    fit += aux
                    last_customer = customer
                self.routes[index_route].fitness = fit
                total_fit += fit
            self.fitness = total_fit

        self.check()
        self.new_2_opt(repetition=2)
        self.change_one(repetition=2)
        self.change_two(repetition=2)
        self.one_elem_two_routes(repetitions=2)
        self.two_elem_two_routes(repetitions=2)
        self.new_2_opt(repetition=1)
        self.change_one(repetition=1)
        self.change_two(repetition=1)
        self.one_elem_two_routes(repetitions=1)
        self.two_elem_two_routes(repetitions=1)
        self.fitness = sum(self.routes[i].fitness for i in range(self.problem.vehicles))
        self.tsp_sr = self.to_tsp_sr()
        self.check()

    def tsp_improved_2_opt(self, act_route: VehicleRoute):
        # tsp_sr = False, Representación de una ruta CVRP [0, ... , 0]
        num_customers = act_route.route.size - 2
        best_solution = np.zeros(num_customers, dtype="int")
        route = np.zeros(num_customers, dtype="int")
        route[:] = act_route.route[1:-1]
        best_solution[:] = act_route.route[1:-1]

        dist = self.problem.distances

        edges = list(itertools.combinations(range(num_customers), 2))
        size_edges = len(edges)
        prev_dist = act_route.fitness
        best_distance = act_route.fitness
        new_distance = self.calc_distance(route, False)
        new_distance += dist[0, route[0]] + dist[route[-1], 0]
        if abs(new_distance - prev_dist) > 1e-4:
            print(new_distance, prev_dist)
            input("Entro mal")

        kk = len(edges)
        if num_customers == 3:
            new_solution = np.array([route[1], route[0], route[2]])
            new_distance = (
                dist[new_solution[0], new_solution[1]]
                + dist[new_solution[1], new_solution[2]]
            )
            new_distance += dist[0, new_solution[0]] + dist[new_solution[-1], 0]
            if new_distance < best_distance:
                best_solution = new_solution
                best_distance = new_distance
            new_solution = np.array([route[0], route[2], route[1]])
            new_distance = (
                dist[new_solution[0], new_solution[1]]
                + dist[new_solution[1], new_solution[2]]
            )
            new_distance += dist[0, new_solution[0]] + dist[new_solution[-1], 0]
            if new_distance < best_distance:
                best_solution = new_solution
                best_distance = new_distance

        elif num_customers > 3:
            kk = int(0.25 * size_edges + 1)
            for k in np.random.choice(range(size_edges), kk, replace=False):
                i, j = edges[k]
                if i == 0 or j + 1 == num_customers:
                    new_solution = np.concatenate(
                        [route[:i], route[i : j + 1][::-1], route[j + 1 :]]
                    )
                    new_distance = self.calc_distance(new_solution, False)
                    new_distance += dist[0, new_solution[0]] + dist[new_solution[-1], 0]
                elif i == j - 1:
                    cost_actual = (
                        dist[route[i - 1], route[i]] + dist[route[j], route[j + 1]]
                    )
                    cost_new = (
                        dist[route[i - 1], route[j]] + dist[route[i], route[j + 1]]
                    )
                    cost_depo = (
                        dist[0, route[0]] + dist[route[-1], 0]
                    )  # Important 0 is the deposit
                    new_solution = np.concatenate(
                        [route[:i], route[i : j + 1][::-1], route[j + 1 :]]
                    )
                    cost_depo -= dist[0, new_solution[0]] + dist[new_solution[-1], 0]
                    new_distance = prev_dist + cost_new - cost_actual + cost_depo
                else:
                    cost_actual = (
                        dist[route[i - 1], route[i]] + dist[route[j], route[j + 1]]
                    )
                    cost_new = (
                        dist[route[i - 1], route[j]] + dist[route[i], route[j + 1]]
                    )
                    cost_depo = (
                        dist[0, route[0]] + dist[route[-1], 0]
                    )  # Important 0 is the deposit
                    new_solution = np.concatenate(
                        [route[:i], route[i : j + 1][::-1], route[j + 1 :]]
                    )
                    # cad += f"*{route}\n>{new_solution}"
                    cost_depo -= dist[0, new_solution[0]] + dist[new_solution[-1], 0]
                    new_distance = prev_dist + cost_new - cost_actual + cost_depo

                if new_distance < best_distance:
                    best_solution = new_solution
                    best_distance = new_distance

        best_solution = np.concatenate([[0], best_solution[:], [0]])
        return best_solution, best_distance

    def tsp_2_hand_opt(self, route):
        num_customers = len(route)
        best_solution = route[:]
        best_distance = self.calc_distance(best_solution, True)
        prev_dist = best_distance
        dist = self.problem.distances

        t = 0
        if num_customers > 3:
            while t < 16 * num_customers:
                i = np.random.randint(num_customers - 1)
                if i + 1 >= num_customers - 1:
                    continue

                j = np.random.randint(low=i + 1, high=num_customers - 1)
                if j + 1 >= num_customers - 1:
                    continue

                k = np.random.randint(low=j + 1, high=num_customers - 1)
                if k + 1 >= num_customers:
                    continue

                l = np.random.randint(low=k + 1, high=num_customers)

                t += 1

                if (
                    l == k + 1
                    or k == j + 1
                    or j == i + 1
                    or l + 1 == num_customers
                    or i == 0
                ):
                    new_solution = np.concatenate(
                        [
                            route[:i],
                            route[k : l + 1][::-1],
                            route[j + 1 : k],
                            route[i : j + 1],
                            route[l + 1 :],
                        ]
                    )

                    new_distance = self.calc_distance(new_solution, True)

                    prev_dist = new_distance

                    if new_distance < best_distance:
                        best_solution = new_solution
                        best_distance = new_distance

                else:
                    cost_actual = (
                        dist[route[i - 1], route[i]]
                        + dist[route[j], route[j + 1]]
                        + dist[route[k - 1], route[k]]
                        + dist[route[l], route[l + 1]]
                    )

                    cost_new = (
                        dist[route[i - 1], route[l]]
                        + dist[route[k], route[j + 1]]
                        + dist[route[k - 1], route[i]]
                        + dist[route[j], route[l + 1]]
                    )

                    new_distance = prev_dist + cost_new - cost_actual

                    prev_dist = new_distance

                    if new_distance < best_distance:
                        best_solution = np.concatenate(
                            [
                                route[:i],
                                route[k : l + 1][::-1],
                                route[j + 1 : k],
                                route[i : j + 1],
                                route[l + 1 :],
                            ]
                        )
                        best_distance = new_distance

        return best_solution

    def initialization_with_repair_from(self, global_route, activate_previa_route=True):
        # Retorna True si es valido, False en caso contrario
        n_route = 0
        # available_capacity = []
        available_route = []
        ind_cust = -1
        # print(f"GRoute {global_route}")
        for customerX in global_route:
            ind_cust += 1
            customer = int(customerX)  # PENDIENTE
            if activate_previa_route:
                input_previous_route = False
                if len(available_route) > 0:
                    n_sel = -1
                    new_cap_sel = -1
                    i_sel = 0
                    for i in range(len(available_route)):
                        n_old = available_route[i]
                        new_cap = (
                            self.routes[n_old].capacity + self.problem.demand[customer]
                        )
                        if new_cap <= self.problem.capacity:
                            # print(f"El {customer} iba para el {n_route}, pero  ingresa a {n_old}. {available_route}")
                            if n_sel == -1:
                                n_sel = n_old
                                new_cap_sel = new_cap
                                i_sel = i
                            if new_cap_sel <= new_cap:
                                n_sel = n_old
                                new_cap_sel = new_cap
                                i_sel = i
                            input_previous_route = True
                    if input_previous_route:
                        last = self.routes[n_sel].route[-1]
                        self.routes[n_sel].route.append(customer)
                        # self.routes[n_sel].fitness += self.problem.distances[last][customer] # 2026
                        self.routes[n_sel].fitness += self.problem.distances[
                            last, customer
                        ]
                        self.routes[n_sel].capacity += self.problem.demand[customer]
                        if self.routes[n_sel].capacity == self.problem.capacity:
                            available_route.pop(i_sel)
                        input_previous_route = True
                    # break
                if input_previous_route:
                    continue
            add_last = True
            if (
                self.routes[n_route].capacity + self.problem.demand[customer]
                > self.problem.capacity
            ):
                if n_route + 1 >= self.problem.vehicles:
                    self.fitness = float("inf")
                    if len(global_route[ind_cust:]) == 0:
                        print(
                            "Error",
                            n_route,
                            customer,
                            global_route[ind_cust:],
                            available_route,
                        )
                    repair = self.repair_instance(
                        n_route, customer, global_route[ind_cust:], available_route
                    )
                    if not repair:
                        return False
                    add_last = False
                else:
                    dif_cap = self.problem.capacity - self.routes[n_route].capacity
                    if dif_cap > 0:
                        available_route = [n_route] + available_route
                    n_route += 1
            if add_last:
                last = self.routes[n_route].route[-1]
                self.routes[n_route].route.append(customer)
                self.routes[n_route].fitness += self.problem.distances[last, customer]
                self.routes[n_route].capacity += self.problem.demand[customer]
        for n_route in range(self.problem.vehicles):
            last = self.routes[n_route].route[-1]
            self.routes[n_route].route.append(0)
            self.routes[n_route].fitness += self.problem.distances[last, 0]
        for i in range(self.problem.vehicles):
            self.routes[i].route = np.array(self.routes[i].route, dtype=int)
        self.fitness = sum(self.routes[i].fitness for i in range(self.problem.vehicles))
        self.tsp_sr = self.to_tsp_sr()
        self.check()
        return True

    def repair_instance(self, n_route, customer, remaining_customers, available_route):

        dif_cap = self.problem.capacity - self.routes[n_route].capacity
        if dif_cap > 0:
            available_route = [n_route] + available_route

        ok_repair = False
        min_route = -1
        min_customer = -1
        min_index = -1
        min_demand = float("inf")

        c = int(remaining_customers[0])
        c_dem = self.problem.demand[c]

        for route in range(self.problem.vehicles - 1, -1, -1):
            list_cust = self.routes[route].route
            capacity = self.routes[route].capacity
            index = 0
            for c_act in list_cust:
                if c_act == 0:
                    continue
                index += 1
                new_cap = capacity + c_dem - self.problem.demand[c_act]
                if (
                    self.problem.demand[c_act] < c_dem
                    and new_cap <= self.problem.capacity
                ):
                    if self.problem.demand[c_act] < min_demand:
                        min_demand = self.problem.demand[c_act]
                        min_route = route
                        min_customer = c_act
                        min_index = index
        if min_route != -1:
            capacity = self.routes[min_route].capacity
            new_cap = capacity + c_dem - self.problem.demand[min_customer]
            last_c = self.routes[min_route].route[min_index - 1]
            self.routes[min_route].fitness -= self.problem.distances[
                last_c, min_customer
            ]
            if min_index + 1 < len(self.routes[min_route].route):
                next_c = self.routes[min_route].route[min_index + 1]
                self.routes[min_route].fitness -= self.problem.distances[
                    min_customer, next_c
                ]
                self.routes[min_route].fitness += self.problem.distances[last_c, next_c]

            self.routes[min_route].route.pop(min_index)  # delete position min_index
            last_c = self.routes[min_route].route[-1]
            self.routes[min_route].route.append(c)  # Add customer c
            self.routes[min_route].fitness += self.problem.distances[last_c, c]

            self.routes[min_route].capacity = new_cap
            c = min_customer
            route = self.problem.vehicles
            while route > 0 and not ok_repair:
                route -= 1
                new_cap = (
                    self.routes[route].capacity + self.problem.demand[min_customer]
                )
                if new_cap <= self.problem.capacity:
                    ok_repair = True
                    last_c = self.routes[route].route[-1]
                    self.routes[route].route.append(min_customer)
                    self.routes[route].capacity = new_cap
                    self.routes[route].fitness += self.problem.distances[
                        last_c, min_customer
                    ]
        return ok_repair

    def pmx(self, other, n_try=1000):
        self.check()
        other.check()
        parent1 = self.tsp_sr
        parent2 = other.tsp_sr
        cont_try = 0
        selection = 1  # This for article
        if selection == 0:
            # Sin corregir
            aux_h1, aux_h2 = self.partially_matched_crossover(parent1, parent2)
            h1 = Solution(self.problem)
            h2 = Solution(self.problem)
            sol_ok = h1.initialization_from(aux_h1)

            if sol_ok:
                min_sol = self if self.fitness <= h1.fitness else h1
            else:
                min_sol = self
            sol_ok = h2.initialization_from(aux_h2)
            if sol_ok:
                min_sol = min_sol if min_sol.fitness <= h2.fitness else h2
            if not min_sol is self:
                self.from_solution(min_sol)

            cont_try += 1
        elif selection == 1:
            # Implementa correcciones.
            aux_h1, aux_h2 = self.partially_matched_crossover(parent1, parent2)
            h1 = SolutionHFA(self.problem)
            h2 = SolutionHFA(self.problem)
            sol_ok = h1.initialization_with_repair_from(aux_h1, False)

            if sol_ok:
                min_sol = self if self.fitness <= h1.fitness else h1
            else:
                min_sol = self
            sol_ok = h2.initialization_with_repair_from(aux_h2, False)
            if sol_ok:
                min_sol = min_sol if min_sol.fitness <= h2.fitness else h2
            if not min_sol is self:
                self.from_solution(min_sol)

            cont_try += 1
        elif selection == 2:
            # Implementa correcciones, pero trata de obtener 2 o 3 descendientes.
            list_solution = []
            while len(list_solution) < 2 and cont_try <= n_try:
                aux_h1, aux_h2 = self.partially_matched_crossover(parent1, parent2)
                h1 = SolutionHFA(self.problem)
                h2 = SolutionHFA(self.problem)

                sol_ok = h1.initialization_with_repair_from(aux_h1, False)

                if sol_ok:
                    list_solution.append(h1)
                sol_ok = h2.initialization_with_repair_from(aux_h2, False)
                if sol_ok:
                    list_solution.append(h2)
                cont_try += 1

            min_sol = self
            for s in list_solution:
                if s.fitness <= min_sol.fitness:
                    min_sol = s

            if not min_sol is self:  # Mover estas lineas...
                self.from_solution(min_sol)

        else:
            while cont_try <= n_try:
                aux_h1, aux_h2 = self.partially_matched_crossover(parent1, parent2)
                h1 = SolutionHFA(self.problem)
                h2 = SolutionHFA(self.problem)
                sol_ok = h1.initialization_with_repair_from(aux_h1, False)

                if sol_ok:
                    sol_ok = h2.initialization_with_repair_from(aux_h2, False)
                if sol_ok:
                    h1.check()
                    h2.check()
                    min_sol = self if self.fitness <= h1.fitness else h1
                    min_sol = min_sol if min_sol.fitness <= h2.fitness else h2
                    if not min_sol is self:
                        self.from_solution(min_sol)
                    break
                cont_try += 1

        if cont_try > n_try:
            print(f".", end="")


# Compatibility alias: there is only one concrete class; legacy internal calls to Solution(...) resolve here.
Solution = SolutionHFA

# Granada, España, 29 de junio de 2026.
