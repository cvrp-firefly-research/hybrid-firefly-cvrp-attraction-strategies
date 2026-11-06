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


from typing import Dict
import numpy as np
import vrplib
from math import atan, pi

def argument(x:float, y: float) -> float:

    argz = None
    if x > 0 and y >= 0:
        argz = atan(y/x)
    elif x < 0:
        argz = pi + atan(y/x)
    elif x > 0 and y < 0:
        argz = 2*pi + atan(y/x)
    elif x == 0 and y > 0:
        argz = pi / 2
    elif x == 0 and y < 0:
        argz = -pi / 2
    return argz


class CVRProblem():
    size: int

    def __init__(self, filename: str) -> None:
        dict_instance: Dict = vrplib.read_instance(filename)
        self.file_name_c: str = filename
        self.file_name: str = dict_instance['name']
        self.name: str = dict_instance['name']  ## Sobra???
        self.vehicles: int = int(self.name.split("-")[2][1:])  # A-n32-k5
        self.comment = dict_instance['comment']
        self.bestKnown: int = int(self.comment.split(":")[-1].split(")")[0])  # (Augerat et al, No of trucks: 5, Optimal value: 784)
        self.type = dict_instance['type']
        self.size = dict_instance['dimension']
        self.edge_weight_type = dict_instance['edge_weight_type']
        self.capacity = dict_instance['capacity']
        self.node_coord = dict_instance['node_coord']
        self.demand = dict_instance['demand']
        self.depot = dict_instance['depot']
        self.edge_weight = dict_instance['edge_weight']
        self.distances = np.round(self.edge_weight)  # Distances Integer
        self.x = self.node_coord[:, 0]  
        self.y = self.node_coord[:, 1]  
        self.r = np.zeros(self.size)
        self.theta = np.zeros(self.size)
        self.to_polar()
        print(self.distances)
        self.sort_cust_polar = np.argsort (self.theta)
        print(self.sort_cust_polar)
        if self.sort_cust_polar[0] != 0:
            index = 0
            while self.sort_cust_polar[index] != 0:
                index += 1
            self.sort_cust_polar = np.concatenate([[0], self.sort_cust_polar[:index], self.sort_cust_polar[index +1:]])
            print(f" {self.sort_cust_polar}")

    def to_polar(self):
        for i in range(1, self.size):
            x = self.x[i] - self.x[0]
            y = self.y[i] - self.y[0]
            self.r[i] = np.sqrt(x**2 + y**2)
            self.theta[i] = argument(x, y)

    def print_customers(self, v):
        print(self.r)
        print(self.file_name)
        print(f"Depositox, (x:{self.x[0]}, y:{self.y[0]})")
        print(len(self.x), len(v))
        print(v)
        for i in  v:
            print(f" {i}, (x:{self.x[i]:3}, y:{self.y[i]:3}) , polar({self.r[i]:4.2f}, {self.theta[i]:4.2f})")

    def __str__(self):
        return f'{self.file_name} size:{self.size} vehicles:{self.vehicles} capacity:{self.capacity} bks:{self.bestKnown}'