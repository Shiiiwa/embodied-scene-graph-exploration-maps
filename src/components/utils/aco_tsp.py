# Code from https://github.com/nishnash54/TSP_ACO

import math
import random
import time as tm
from tqdm import tqdm


class SolveTSPUsingACO:
    class Edge:
        def __init__(self, a, b, weight, initial_pheromone):
            self.a = a
            self.b = b
            if weight == 0:
                weight = 1e-10
            self.weight = weight
            self.pheromone = initial_pheromone

    class Ant:
        def __init__(self, alpha, beta, num_nodes, edges):
            self.alpha = alpha
            self.beta = beta
            self.num_nodes = num_nodes
            self.edges = edges
            self.tour = None
            self.distance = 0.0

        def _select_node(self):
            current = self.tour[-1]
            unvisited = [n for n in range(self.num_nodes) if n not in self.tour]
            if not unvisited:
                return None

            scores = []
            total = 0.0
            for u in unvisited:
                e = self.edges[current][u]
                tau = max(e.pheromone, 1e-12) ** self.alpha
                eta = (1.0 / max(e.weight, 1e-12)) ** self.beta
                s = tau * eta
                scores.append((u, s))
                total += s

            if total <= 0.0 or not scores:
                # fallback: choose random
                return random.choice(unvisited)

            # Roulette-Wheel
            r = random.uniform(0.0, total)
            acc = 0.0
            for u, s in scores:
                acc += s
                if acc >= r:
                    return u
            # fallnack
            return scores[-1][0]

        def find_tour(self):
            self.tour = [random.randint(0, self.num_nodes - 1)]
            while len(self.tour) < self.num_nodes:
                nxt = self._select_node()
                if nxt is None or nxt in self.tour:
                    remaining = [n for n in range(self.num_nodes) if n not in self.tour]
                    if not remaining:
                        break
                    nxt = random.choice(remaining)
                self.tour.append(nxt)
            return self.tour

        def get_distance(self):
            self.distance = 0.0
            for i in range(self.num_nodes):
                self.distance += self.edges[self.tour[i]][self.tour[(i + 1) % self.num_nodes]].weight
            return self.distance

    def __init__(
        self,
        mode="ACS",
        colony_size=10,
        elitist_weight=1.0,
        min_scaling_factor=0.001,
        alpha=1.0,
        beta=3.0,
        rho=0.1,
        pheromone_deposit_weight=1.0,
        initial_pheromone=1.0,
        steps=100,
        nodes=None,
        labels=None,
    ):
        self.mode = mode
        self.colony_size = colony_size
        self.elitist_weight = elitist_weight
        self.min_scaling_factor = min_scaling_factor
        self.rho = rho
        self.pheromone_deposit_weight = pheromone_deposit_weight
        self.steps = steps
        self.num_nodes = len(nodes)
        self.nodes = nodes
        if labels is not None:
            self.labels = labels
        else:
            self.labels = range(1, self.num_nodes + 1)
        self.edges = [[None] * self.num_nodes for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.edges[i][j] = self.edges[j][i] = self.Edge(
                    i,
                    j,
                    math.sqrt(pow(self.nodes[i][0] - self.nodes[j][0], 2.0) + pow(self.nodes[i][1] - self.nodes[j][1], 2.0)),
                    initial_pheromone,
                )
        self.ants = [self.Ant(alpha, beta, self.num_nodes, self.edges) for _ in range(self.colony_size)]
        self.global_best_tour = None
        self.global_best_distance = float("inf")

    def _add_pheromone(self, tour, distance, weight=1.0):
        pheromone_to_add = self.pheromone_deposit_weight / distance
        for i in range(self.num_nodes):
            self.edges[tour[i]][tour[(i + 1) % self.num_nodes]].pheromone += weight * pheromone_to_add

    def _acs(self):
        for step in range(self.steps):
            for ant in self.ants:
                # Amout of pheromone increment is constant.
                self._add_pheromone(ant.find_tour(), ant.get_distance())
                if ant.distance < self.global_best_distance:
                    self.global_best_tour = ant.tour
                    self.global_best_distance = ant.distance
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= 1.0 - self.rho

    def _elitist(self):
        for step in range(self.steps):
            for ant in self.ants:
                self._add_pheromone(ant.find_tour(), ant.get_distance())
                if ant.distance < self.global_best_distance:
                    self.global_best_tour = ant.tour
                    self.global_best_distance = ant.distance
            # Amout of pheromone depends upon the quality of the solution.
            self._add_pheromone(self.global_best_tour, self.global_best_distance, weight=self.elitist_weight)
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= 1.0 - self.rho

    def _max_min(self):
        for step in range(self.steps):
            iteration_best_tour = None
            iteration_best_distance = float("inf")
            for ant in self.ants:
                ant.find_tour()
                if ant.get_distance() < iteration_best_distance:
                    iteration_best_tour = ant.tour
                    iteration_best_distance = ant.distance
            if float(step + 1) / float(self.steps) <= 0.75:
                # Initially higher weight gradually reduced will 75% completion
                self._add_pheromone(iteration_best_tour, iteration_best_distance)
                max_pheromone = self.pheromone_deposit_weight / iteration_best_distance
            else:
                # Last 25% weights based on quality of route after comparing with the global best tour.
                if iteration_best_distance < self.global_best_distance:
                    self.global_best_tour = iteration_best_tour
                    self.global_best_distance = iteration_best_distance
                self._add_pheromone(self.global_best_tour, self.global_best_distance)
                max_pheromone = self.pheromone_deposit_weight / self.global_best_distance
            min_pheromone = max_pheromone * self.min_scaling_factor

            # Setting edge weights to min and max phermones
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= 1.0 - self.rho
                    if self.edges[i][j].pheromone > max_pheromone:
                        self.edges[i][j].pheromone = max_pheromone
                    elif self.edges[i][j].pheromone < min_pheromone:
                        self.edges[i][j].pheromone = min_pheromone

    def run(self):
        start = tm.time()

        for ant in self.ants:
            ant.find_tour()
            d = ant.get_distance()
            if d < self.global_best_distance:
                self.global_best_distance = d
                self.global_best_tour = ant.tour[:]

        if self.mode == "ACS":
            self._acs()
        elif self.mode == "Elitist":
            self._elitist()
        else:
            self._max_min()

        return tm.time() - start, self.global_best_distance


if __name__ == "__main__":
    _colony_size = 5
    _steps = 50

    f = open("./out.csv", "w")

    f.write(",".join(["Iteration", "ACS_time", "ACS_dist", "Elitist_time", "Elitist_dist", "MaxMin_time", "MaxMin_dist"]))
    f.write("\n")

    # Analyse and compare the various modes
    for i in range(10):

        print("Iter: ", i + 1)

        for j in tqdm(range(20)):
            _nodes = [(random.uniform(-400, 400), random.uniform(-400, 400)) for _ in range(0, 10 * (i + 1))]
            acs = SolveTSPUsingACO(mode="ACS", colony_size=_colony_size, steps=_steps, nodes=_nodes)
            elitist = SolveTSPUsingACO(mode="Elitist", colony_size=_colony_size, steps=_steps, nodes=_nodes)
            max_min = SolveTSPUsingACO(mode="MaxMin", colony_size=_colony_size, steps=_steps, nodes=_nodes)

            time, dist = acs.run()
            f.write(str(i + 1) + "," + str(time) + "," + str(dist) + ",")
            time, dist = elitist.run()
            f.write(str(time) + "," + str(dist) + ",")
            time, dist = max_min.run()
            f.write(str(time) + "," + str(dist) + "\n")

    f.close()
