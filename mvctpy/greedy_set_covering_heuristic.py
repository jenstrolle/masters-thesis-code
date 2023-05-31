import numpy as np
from utils import *

class set_covering_solver:
    """
    Class for handling and solving set covering problems.
    """
    def __init__(self, n, V, W, used, covering, distances, args, first):
        self.n = n
        self.W = W
        self.V = V
        self.used = used
        self.C = covering.copy()
        self.D = distances
        self.random = args.alpha
        if first:
            self.random = 1
        self.costs = self.cost_function(V, distances)
        pass

    def cost_function(self, V, distances):
        """
        Calculates the cost of each node in V.
        """
        v_size = len(V)
        costs = np.zeros(v_size)
        v_nodes_distances = distances[:v_size, :v_size]

        for i, dists in enumerate(v_nodes_distances):
            costs[i] = np.mean(np.sort(dists)[1:2+1])

        return costs

    def remove_previously_covered(self, U, S):
        """
        Removes the nodes which are covered by the required nodes.
        """
        for required in self.used:
            # if covered vertex is the depot, add to sol and continue
            if required == 0:
                S.add(required)
                continue

            # covering is the covering matrix
            already_covered_by_required = set(np.where(self.C[:, required] == 1)[0])

            if already_covered_by_required is not set():
                S.add(required)
                U = U.difference(already_covered_by_required)
                self.C[list(already_covered_by_required), :] = np.zeros((len(self.V)))
        return U, S

    def solver(self):

        # N is set of nodes to consider
        N = set(range(len(self.V)))

        # U is the set of uncovered vertices
        U = set(range(self.n-len(self.V)))

        # S is the solution set
        S = set()


        # removes required nodes from consideration, and the nodes which are
        # covered by the required nodes
        U, S = self.remove_previously_covered(U, S)
        N = N.difference(S)

        row_sums = np.sum(self.C, axis=0)
        orig = N.copy()

        # Remove any nodes from N which do not cover any from U
        non_covered = set(np.where(row_sums == 0)[0])
        N = N.difference(non_covered)
        orig = N.copy()


        non_empty = np.where(np.sum(self.C, axis=1) > 0, 1, 0)

        # removes required nodes from consideration, and the nodes which are
        # covered by the required nodes
        total = [list(S)]
        self.costs = np.array([cost if i in N else np.inf for i, cost in
                               enumerate(self.costs)])

        if sum(row_sums) == 0:
            return list(S), total, orig, non_empty

        while True:
            J = np.array(list(N.difference(S)))

            # might divide by 0 - just set ratio to 0
            ratios = list(zip(self.costs[J]/row_sums[J], J))

            cond = min(ratios)[0]

            if self.random:
                cond_max = max(ratios)[0]
                cond = cond + self.random*(cond_max - cond)


            R = np.array([j for cr_j, j in ratios if cr_j <= cond])

            if self.random:
                k = np.random.choice(R)
            else:
                k = R[0]

            S.add(k)

            J = list(N.difference(S))
            U_prime = set([i for i in U if self.C[i, k] == 1])

            U = U.difference(U_prime)
            total.append(list(S))

            if len(U) <= 0:
                break

            row_sums[J] -= np.sum(self.C[:, J][list(U_prime), :], axis=0)

            # make sure we do not have row sums below 0 (happens if multiple v_h
            # cover the same v_l)
            row_sums = np.where(row_sums < 0, 0, row_sums)

            remove = set(np.where(row_sums == 0)[0])

            N = N.difference(remove)


        return list(S), total, orig, non_empty
