import numpy as np
import math
from .utils import *
import matplotlib.pyplot as plt

def choose_heuristic(heuristics):
    """
    Chooses a heuristic to apply based on the weights of each heuristic.

    Parameters:
        heuristics (dict): dictionary of heuristics and their weights

    Outputs:
        heuristic (str): heuristic to apply
    """

    w = np.zeros(len(heuristics))
    for i, heur in enumerate(heuristics.values()):
        if heur['weight'] == np.nan:
            w[i] = 0
        else:
            w[i] = heur['weight']

    probs = w/np.sum(w)

    return np.random.choice(a=list(heuristics), p=probs)

def update_weights(heuristics, args):
    """
    Updates the weights of each heuristic based on the scores of each
    heuristic.

    Parameters:
        heuristics (dict): dictionary of heuristics and their weights

        args (argsparse.NameSpace): command line arguments

    Outputs:
        heuristics (dict): dictionary of heuristics and their weights with updated
                    weights
    """

    r = args.r

    for heur in heuristics:
        w = heuristics[heur]['weight']
        if w == np.nan:
            w = 0
        s = heuristics[heur]['score']['value']
        att = heuristics[heur]['score']['attempts']
        if att == 0:
            heuristics[heur]['weight'] = w*(1-r)

        else:
            heuristics[heur]['weight'] = w*(1-r) + r*(s/att)

def update_score(destroy_heurs, repair_heurs, current_destroy, current_repair, score_type, args_dict):
    """
    Updates the score of each heuristic based on the type of score.

    Parameters:
        destroy_heurs (dict): dictionary of destroy heuristics and their scores

        repair_heurs (dict): dictionary of repair heuristics and their scores

        current_destroy (str): current destroy heuristic

        current_repair (str): current repair heuristic

        score_type (str): type of score to update

        args_dict (dict): dictionary of command line arguments

    Outputs:
        destroy_heurs (dict): dictionary of destroy heuristics and their scores with updated
                    scores

        repair_heurs (dict): dictionary of repair heuristics and their scores with
                    updated scores
    """

    score_value = args_dict[score_type]
    destroy_heurs[current_destroy]['score']['value'] += score_value
    repair_heurs[current_repair]['score']['value'] += score_value


def accept_solution(current_obj, new_obj, temp):
    """
    Accepts a solution based on the simulated annealing acceptance criterion.

    Parameters:
        current_obj (float): objective value of current solution

        new_obj (float): objective value of new solution

        temp (float): current temperature

    Outputs:
        accept (bool): boolean indicating whether to accept the new solution

        accept_type (str): type of acceptance

    """

    better_than_old = (new_obj < current_obj)
    if better_than_old:
        return True, 's2'
    else:
        diff_in_obj = new_obj - current_obj

        try:
            p = math.log(np.random.rand())
            diff = -diff_in_obj/temp
            sa_accept = p <= diff

        except FloatingPointError:
            sa_accept = False

        if sa_accept:
            return True, 's3'

    return False, None

def update_temp(temp, args):
    """
    Updates the temperature based on the cooling schedule.

    Parameters:
        temp (float): current temperature

        args (argsparse.NameSpace): command line arguments

    Outputs:
        temp (float): updated temperature

    """
    c = args.c
    return temp * c


def plot_results(instance, start_sol, best_sol, objs, best_objs,
                 weights_1, weights_2):
    """
    Plots the results of the ALNS algorithm.
    """
    nodes = instance.nodes
    t_size = instance.T_size
    v_size = instance.V_size

    t = nodes[:t_size]
    W = nodes[v_size:]
    W_coords = get_coords(W)
    t_coords = get_coords(t)

    # logic for plotting the initial solution and the best solution
    fig, ax = plt.subplots(1, 2)
    for i, sol in enumerate([start_sol, best_sol]):
        ax[i].axis('equal')
        ax[i].scatter(*get_coords(nodes[t_size:v_size]), label='V \\ T nodes')
        ax[i].scatter(*W_coords, label='W nodes')
        sol_index = flatten(sol)
        sol_nodes = nodes[sol_index]

        sol_coords = get_coords(sol_nodes)
        ax[i].scatter(*sol_coords, label='Set cover')
        ax[i].scatter(*t_coords, label='T nodes')

        for w in sol_nodes:
            c = plt.Circle(w, radius=instance.covering_distance, fc='none', ec='black')
            ax[i].add_patch(c)

        for r in sol:
            r = [0] + r + [0]
            covering_v_routes = get_coords(nodes[r])
            ax[i].plot(*covering_v_routes)

        ax[i].legend()
    plt.show()

    # logic for plotting the evolution of the weights of the heuristics
    heur_names = [['Greedy Insertion',
                '2-regret',
                '3-regret',
                '4-regret',
                '$m$-regret',
                'Random Insertion'],
                ['Shaw Removal',
                'Random Removal',
                'Worst Removal',
                'Least Covering Removal']]

    for j, weights in enumerate([weights_1, weights_2]):
        fig, ax = plt.subplots(1, 1)
        for i, w in enumerate(weights):
            ax.plot(np.arange(len(weights[w])), weights[w],
            marker='+', label=heur_names[j][i])

        ax.legend()
        ax.set_xlabel('Segment')
        ax.set_ylabel('Weight')
        ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
        if j == 0:
            title = 'Evolution in Weights of Repair-Heuristics'
        else:
            title = 'Evolution in Weights of Destruction-Heuristics'
        ax.set_title(title)
        #plt.savefig(f'{title}.pdf')

    plt.show()

    # logic for plotting the evolution of the objective values
    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(len(objs)), objs, color='black',
            label='All Objective Values')
    ax.plot(np.arange(len(best_objs)), best_objs, color='red',
            label='Best Objective Values')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
    ax.set_title('The Current and Best Objective Value in Each Iteration')
    ax.legend()

    #plt.savefig(f'objs.pdf')
    plt.show()
