from mvctpy.utils import *
import numpy as np
import random

def shaw(gamma, S, S_nodes, covering, dists, args):
    """
    Applies the Shaw Removal Heuristic to a solution S.

    Parameters:
        gamma: number of nodes to remove
        S: current solution
        S_nodes: nodes in current solution
        covering: matrix of nodes covered by each route
        dists: distance matrix
        args: command line arguments
            - includes delta, the randomness
              parameter for the shaw removal heuristic

    Outputs:
        F: set of nodes to remove
    """

    delta = args.delta
    R = dists[S_nodes, :][:, S_nodes]
    S = [i for i in range(len(S_nodes))]
    F = []
    sigma = np.random.choice(S)
    # append the element which is popped from S
    F.append(S.pop(S.index(sigma)))

    while len(F) < gamma:
        sigma_i = np.random.choice(F)
        related_to_i = sorted([(R[sigma_i, i], i) for i in S], key=lambda x:
                              x[0], reverse=True)
        y = np.random.random()
        idx = int(np.floor((y**delta)*len(related_to_i)))

        sigma_j = related_to_i[idx][1]
        F.append(S.pop(S.index(sigma_j)))

    F = simple_nodes_to_actual_indices(F, S_nodes)

    return F


def worst_removal(gamma, S, S_nodes, covering, dists, args):
    """
    Applies the Worst Removal Heuristic to a solution S.

    Parameters:
        gamma: number of nodes to remove
        S: current solution
        S_nodes: nodes in current solution
        covering: matrix of nodes covered by each route
        dists: distance matrix
        args: command line arguments
            - includes lambd, the randomness parameter 
              for the worst removal heuristic
    
    Outputs:
        F: set of nodes to remove  
    """

    lambd = args.lambd
    F = []
    row_sums = np.sum(covering, axis=0)
    S_temp = S_nodes.copy()
    temp_sol = S.copy()
    while len(F) < gamma:
        cost_list = []
        edge_sol .alns_helper_func= [route_to_edges([0]+route+[0]) for route in temp_sol]
        total_obj = objective(edge_sol, dists)
        for s in S_temp:
            _S = [[n for n in route if n != s] for route in temp_sol]
            r = [idx for idx, route in enumerate(temp_sol) if s in route]

            _edge_sol = [route_to_edges([0]+route+[0]) for route in _S]

            if r == []:
                r = [0]
            cost_list.append((total_obj-objective(_edge_sol, dists), s, (r[0], _S[r[0]])))
        cost_list = sorted(cost_list, key=lambda x: x[0], reverse=True)

        y = np.random.random()
        idx = int(np.floor((y**lambd)*len(cost_list)))
        idx_to_pop = cost_list[idx][1]
        temp_sol[cost_list[idx][2][0]] = cost_list[idx][2][1]
        F.append(S_temp.pop(S_temp.index(idx_to_pop)))
    return F

def random_removal(gamma, S, S_nodes, covering, dists, args):
    """
    Applies the Random Removal Heuristic to a solution S.

    Parameters:
        gamma: number of nodes to remove
        S: current solution
        S_nodes: nodes in current solution
        covering: matrix of nodes covered by each route
        dists: distance matrix
        args: command line arguments
    
    Outputs:
        F: set of nodes to remove  
    """

    F = random.sample(S_nodes, gamma)

    return F

def least_covering(gamma, S, S_nodes, covering, dists, args):
    """
    Applies the Least Covering Removal Heuristic to a solution S.

    Parameters:
        gamma: number of nodes to remove
        S: current solution
        S_nodes: nodes in current solution
        covering: matrix of nodes covered by each route
        dists: distance matrix
        args: command line arguments
            - includes eta, the randomness parameter
              for the least covering removal heuristic
    
    outputs:   
        F: set of nodes to remove
    """

    eta = args.eta
    row_sums = np.sum(covering, axis=0)
    F = []
    S_temp = S_nodes.copy()

    while len(F) < gamma:
        considered = row_sums[S_temp]
        considered_index = sorted(list(zip(considered, S_temp)), key=lambda x:
                                  x[0])
        y = np.random.random()
        idx = int(np.floor((y**eta)*len(S_temp)))
        idx_to_pop = considered_index[idx][1]
        F.append(S_temp.pop(S_temp.index(idx_to_pop)))

    return F