#!/usr/bin/python3
from .alns_helper_functions import choose_heuristic, update_weights, update_score, update_temp, accept_solution, plot_results
from .destroy_heuristics import random_removal, shaw, worst_removal, least_covering
from .repair_heuristics import greedy_insertion, k_regret, random_insertion

from .greedy_set_covering_heuristic import set_covering_solver
from .tsp_to_mvctp import tsp_to_mvctp_instance as tsp

from .utils import *

import argparse
import math
import numpy as np

def ALNS(instance, params):
    """
    Applies the Adaptive Large Neighborhood Search algorithm to a given
    instance.

    Parameters:
        instance (tsp_to_mvctp_instance): instance object

        params (argparse.NameSpace): command line arguments

    Outputs:
        best_obj (float): objective value of best solution found

    """
    # initialize dict of parameters
    args_dict = vars(params)

    # initialize instance objects
    V, W, T, D, B, _ = instance.out()
    length_info = instance.length_info
    N = instance.n
    v_size = instance.V_size

    # initialize covering matrix
    W_V_covering = B[v_size:, :v_size]

    # set stop
    STOP = False
    i = 0
    params.noise = False

    # initialize feasible solution
    set_cover_object = set_covering_solver(N, V, W, T, W_V_covering, D,
                                           params, first=True)

    set_cover, _, _, _ = set_cover_object.solver()

    cur_sol = k_regret([[]], set_cover[1:], D, length_info, 2, params)

    start_sol = cur_sol
    cur_edge_sol = [route_to_edges([0]+route+[0]) for route in cur_sol]
    cur_obj = objective(cur_edge_sol, D)

    w = params.w
    temp = -(cur_obj*w)/(math.log(1/2))

    cur_nodes = flatten(cur_sol)
    best_sol = cur_sol
    best_obj = objective(cur_edge_sol, D)
    m = len(cur_sol)

    # initialize destroy and repair heuristics
    destroy_heuristics = {'shaw': {'func':shaw, 'weight':1,
                                'score':{'value':0, 'attempts':0}},
                       'random':{'func':random_removal, 'weight':1,
                                 'score':{'value':0, 'attempts':0}},
                       'worst':{'func':worst_removal, 'weight':1,
                                'score':{'value':0, 'attempts':0}},
                       'least_covering':{'func':least_covering, 'weight':1,
                                         'score':{'value':0, 'attempts':0}}}

    repair_heuristics = {'greedy': {'func':greedy_insertion, 'weight':1,
                                    'score':{'value':0, 'attempts':0}},
                         '2-regret':{'func':k_regret, 'weight':1,
                                     'score':{'value':0, 'attempts':0}, 'q':2},
                         '3-regret':{'func':k_regret, 'weight':1,
                                     'score':{'value':0, 'attempts':0}, 'q':3},
                         '4-regret':{'func':k_regret, 'weight':1,
                                     'score':{'value':0, 'attempts':0}, 'q':4},
                         'm-regret':{'func':k_regret, 'weight':1,
                                     'score':{'value':0, 'attempts':0}, 'q':m},
                         'random':{'func':random_insertion, 'weight':1,
                                   'score':{'value':0, 'attempts':0}}}

    # initialize dict of all weights for each heuristic for plotting purposes
    all_destroy_weights = {'shaw':[1], 'random':[1], 'worst':[1], 'least_covering':[1]}
    all_repair_weights = {'greedy':[1], '2-regret':[1], '3-regret':[1],
                          '4-regret':[1], 'm-regret':[1], 'random':[1]}

    # initialize lists for plotting purposes
    objs = []
    best_objs = []

    # start ALNS loop
    while params.phi >= i:

        # append to lists for plotting purposes if plotting is enabled
        if params.plot:
            objs.append(cur_obj)
            best_objs.append(best_obj)

        # determine the degree of destruction in the current iteration
        gamma = int(np.random.uniform(0.3*len(cur_nodes), params.epsilon*len(cur_nodes)))

        # weight adjustment step (update ensures no update in first itear)
        if i % params.tau == 0:
            update_weights(destroy_heuristics, params)
            for destroy in list(destroy_heuristics):
                if params.plot:
                    all_destroy_weights[destroy].append(destroy_heuristics[destroy]['weight'])

                destroy_heuristics[destroy]['score']['value'] = 0
                destroy_heuristics[destroy]['score']['attempts'] = 0

            update_weights(repair_heuristics, params)
            for repair in list(repair_heuristics):
                if params.plot:
                    all_repair_weights[repair].append(repair_heuristics[repair]['weight'])

                repair_heuristics[repair]['score']['value'] = 0
                repair_heuristics[repair]['score']['attempts'] = 0

        # choose destroy and repair heuristics and update attempts
        cur_destroy = choose_heuristic(destroy_heuristics)
        cur_repair = choose_heuristic(repair_heuristics)
        destroy_heuristics[cur_destroy]['score']['attempts'] += 1
        repair_heuristics[cur_repair]['score']['attempts'] += 1

        # set value of q
        if cur_repair[2:] == 'regret':
            q = repair_heuristics[cur_repair]['q']
        else:
            q = 0

        # apply destroy heuristic
        cur_destroy_result = destroy_heuristics[cur_destroy]['func'](gamma, cur_sol, cur_nodes,
                                                            W_V_covering, D,
                                                            params)


        # remove destroyed nodes from solution
        cur_destroy_sol = routes_with_destruction_removed(cur_destroy_result, cur_sol)
        after_destroy = [0] + flatten(cur_destroy_sol)

        # apply greedy set covering heuristic to destroyed solution
        new_sc_object = set_covering_solver(N, V, W, after_destroy,
                                            W_V_covering, D, params,
                                            first=False)

        new_sc_sol, _, _, _ = new_sc_object.solver()


        # get list of nodes to insert into destroyed solution
        insertion_nodes = list(set(new_sc_sol).difference(set(after_destroy)))

        # apply repair heuristic
        new_sol = repair_heuristics[cur_repair]['func'](cur_destroy_sol,
                                                        insertion_nodes, D,
                                                        length_info, q,
                                                        params)

        # compute objective of new solution
        sol_as_edges = [route_to_edges([0]+s+[0]) for s in new_sol]
        new_obj = objective(sol_as_edges, D)
        new_nodes = flatten(new_sol)

        # apply the route combination step
        if len(new_sol) > 1:
            temp_sol, temp_nodes = new_sol[1:].copy(), new_sol[0].copy()
            vehicle_minimization = repair_heuristics[cur_repair]['func'](temp_sol,
                                                        temp_nodes, D,
                                                        length_info, q,
                                                        params)

            vehicle_min_edges = [route_to_edges([0]+s+[0]) for s in vehicle_minimization]
            vehicle_min_obj = objective(vehicle_min_edges, D)

            if len(vehicle_minimization) <= len(new_sol) or vehicle_min_obj < new_obj:
                new_sol = vehicle_minimization
                new_nodes = flatten(new_sol)
                sol_as_edges = vehicle_min_edges
                new_obj = vehicle_min_obj

        # update best solution if new solution is best found so far
        if new_obj < best_obj:
            update_score(destroy_heuristics, repair_heuristics, cur_destroy,
                         cur_repair, 's1', args_dict)

            cur_sol = new_sol
            best_obj, best_sol = new_obj, new_sol
            cur_obj = best_obj
            cur_nodes = flatten(cur_sol)

            temp = update_temp(temp, params)
            i += 1

            if params.verbose:
                print(f'new best solution found: {best_sol} with objective {best_obj}')

            continue

        # decide acceptance of new solution if it is not the best found so far
        accept, accept_type = accept_solution(cur_obj, new_obj, temp)

        if accept:
            update_score(destroy_heuristics, repair_heuristics, cur_destroy,
                        cur_repair, accept_type, args_dict)

            cur_sol = new_sol
            cur_nodes = new_nodes
            cur_obj = new_obj

        # update temperature and iteration count
        temp = update_temp(temp, params)
        i += 1

    if params.plot:
        plot_results(instance, start_sol, best_sol, objs, best_objs,
                     all_repair_weights, all_destroy_weights)

        print(f'best objective found: {best_obj}\n',
              f'on routes {best_sol}')
        return best_obj

    else:
        print(f'best objective found: {best_obj}\n',
              f'on routes {best_sol}')
        return best_obj


if __name__ == "__main__":
    desc = "Solving MVCTP with ALNS algorithm"
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--instance", help="instance to attept to solve", required=True)
    ap.add_argument("--delta", type=int, help="randomness in shaw removal", required=True)
    ap.add_argument("--lambd", type=int, help="randomness in worst removal", required=True)
    ap.add_argument("--eta", type=int, help="randomness in least covering removal", required=True)
    ap.add_argument("--alpha", type=float, help="randomness in set covering solver", default=1)
    ap.add_argument("--r", type=float, help="weight adjustment change rate", required=True)
    ap.add_argument("--s1", type=int, help="score for finding new global best solution", required=True)
    ap.add_argument("--s2", type=int, help="score for finding an improving solution", required=True)
    ap.add_argument("--s3", type=int, help="score for finding an accepted non-improving solution", required=True)
    ap.add_argument("--epsilon", type=float, help="factor determining upper limit of nodes removed in destroy step", required=True)
    ap.add_argument("--w", type=float, help="parameter decicing quality of solution accepted with probability 1/2 in SA acceptance scheme", required=True)
    ap.add_argument("--c", type=float, help="cooling factor in SA", required=True)
    ap.add_argument("--phi", type=int, help="number of ALNS iterations to run",
                    default=25000)
    ap.add_argument("--tau", type=int, help="number of iterations in a period",
                    default=100)
    ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    ap.add_argument("-plot", "--plot", help="plot results", action="store_true", required=False, default=True)

    args = ap.parse_args()

    ALNS(tsp(args.instance), args)
