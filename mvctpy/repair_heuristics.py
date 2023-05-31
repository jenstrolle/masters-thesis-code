from mvctpy.utils import *
import numpy as np
import random

def regret_value(costs, x_ik, q):
    """
    Computes the regret value for each node in a solution.

    Parameters:
        costs: matrix of costs for each node in each route
        x_ik: matrix of indices of the q lowest costs for each node
        q: number of lowest costs to consider
    
    Outputs:
        regrets: vector of regret values for each node
    """

    least_cost = costs[:, 0][:, np.newaxis]
    regrets = np.sum(costs[:, 1:q+1] - least_cost, axis=1)[:, np.newaxis]
    return regrets

def length_checker(routes, nodes, dists, max_length, args):
    """
    Checks if a set of nodes can be inserted into a set of routes without
    violating the maximum route length constraint.

    Parameters:
        routes: set of routes
        nodes: set of nodes to insert
        dists: distance matrix
        max_length: maximum route length
        args: command line arguments
    Outputs:
        lengths: matrix of lengths of routes with nodes inserted
        lengths_bool: matrix of booleans indicating whether a route is too long
        length_info: list of tuples containing information about the insertion
                        of each node into each route
    """

    lengths = np.zeros((len(nodes), len(routes)))
    length_info = []

    for i, node in enumerate(nodes):
        info = []
        for j, route in enumerate(routes):
            obj, _, route = add_node_to_route(node, route, dists, args)
            lengths[i, j] = obj
            info.append((obj, i, j, route))
        length_info.append(info)
    lengths_bool = np.where(lengths <= max_length, 1, 0)
    return lengths, lengths_bool, length_info

def k_regret(old_solution, nodes_to_insert, dists, instance_length_info, regret_num, args):
    """
    Applies the k-regret insertion heuristic to a destroyed solution using
    the new nodes from the set covering heuristic.

    Parameters:
        old_solution: destroyed solution
        nodes_to_insert: set of nodes to insert
        dists: distance matrix
        instance_length_info: information about the route constraints
        regret_num: regret value to be applied
        args: command line arguments
    
    Outputs:
        solution: new solution
    """

    q = regret_num
    length_type, max_length = instance_length_info
    solution = old_solution.copy()
    too_long = []
    check = False
    while nodes_to_insert:
        if length_type == 'vertices':
            route_lengths = np.array([len(r) for r in solution])
            route_too_long = np.where(route_lengths + 1 > max_length)[0]
            if len(route_too_long):
                route_too_long = list(route_too_long)
                for idx in sorted(route_too_long, reverse=True):
                    too_long.append(solution[idx])
                    del solution[idx]
        else:
            lengths, lengths_bool, length_info = length_checker(solution, nodes_to_insert,
                                                 dists, max_length, args)

            routes_too_long_dists = np.sum(lengths_bool, axis=1)
            permissable_sum = np.sum(routes_too_long_dists)
            check = False

            if permissable_sum == 0:
                solution.append([])

            elif any(routes_too_long_dists > len(solution) - q + 1): #or permissable_sum != 0:
                non_zeros = np.where(routes_too_long_dists > 0)[0]
                lowest = non_zeros[routes_too_long_dists[non_zeros].argmin()]

                changes = min(length_info[lowest], key=lambda x: x[0])

                _, placed_node_index, changed_route_index, new_route = changes
                solution[changed_route_index] = new_route
                del nodes_to_insert[placed_node_index]
                continue

            else:
                x = {}
                info = []
                costs = np.array([])
                nodes_cost = {}
                nodes_info = {}
                for i, k in np.argwhere(lengths_bool == 1):
                    obj, _, route = add_node_to_route(nodes_to_insert[i],
                                                           solution[k], dists)
                    if i not in nodes_cost:
                        nodes_cost[i] = [obj]
                        nodes_info[i] = [[obj, i, k, route]]

                    else:
                        nodes_cost[i] = nodes_cost[i] + [obj]
                        nodes_info[i] = nodes_info[i] + [[obj, i, k, route]]
                        if len(nodes_cost[i]) < q:
                            pass
                        else:
                            x[i] = np.argsort(np.array(nodes_cost[i]))[:q]

                for i in nodes_cost:
                    nodes_cost[i] = np.sort(np.array(nodes_cost[i]))

                regrets = {}
                for i in nodes_cost:
                    regrets[i] = np.sum(nodes_cost[i][1:q+1] - nodes_cost[i][0])

                max_regret = max(regrets, key=regrets.get)

                changes = min(nodes_info[max_regret], key=lambda x: x[0])

                _, placed_node_index, changed_route_index, new_route = changes
                solution[changed_route_index] = new_route
                del nodes_to_insert[placed_node_index]
                continue



        if len(nodes_to_insert) and solution == []:
            solution.append([])

        x = np.zeros((len(nodes_to_insert), q))
        total_info = []
        total_costs = np.zeros((len(nodes_to_insert), len(solution)))

        for i, node in enumerate(nodes_to_insert):
            info = []
            i_costs = np.array([])
            for k, route in enumerate(solution):
                obj, _, route = add_node_to_route(node, route, dists)
                i_costs = np.append(i_costs, np.array([obj]))
                info.append((obj, i, k, route))
            if len(i_costs) < q:
                pass
            else:
                x[i] = np.argsort(i_costs)[:q]
            total_costs[i] = i_costs
            total_info.append(info)
        sorted_costs = np.sort(total_costs)
        max_regrets = np.argmax(regret_value(sorted_costs, x, q))


        changes = min(total_info[max_regrets], key=lambda x: x[0])

        _, placed_node_index, changed_route_index, new_route = changes
        solution[changed_route_index] = new_route
        del nodes_to_insert[placed_node_index]

    if len(too_long):
        solution = solution + too_long

    return solution

def random_insertion(old_solution, nodes_to_insert, dists, length_info,
                     regret_num, args):
    """
    Applies the random insertion heuristic to a destroyed solution using 
    the new nodes from the set covering heuristic.

    Parameters:
        old_solution: destroyed solution
        nodes_to_insert: set of nodes to insert
        dists: distance matrix
        instance_length_info: information about the route constraints
        regret_num: regret value to be applied
        args: command line arguments
    
    Outputs:
        solution: new solution
    """

    length_type, max_length = length_info
    solution = old_solution.copy()


    for node in nodes_to_insert:
        if length_type == 'vertices':
            candidates = [route for route in solution if len(route) < max_length]
        else:
            _, candidate_matrix, _ = length_checker(solution, nodes_to_insert, dists, max_length, args)
            permissable = np.where(np.all(candidate_matrix, axis=0))[0]

            if permissable.size == 0:
                candidates = False
            else:
                candidates = [solution[i] for i in permissable]

        if candidates:
            route = random.choice(candidates)
        else:
            route = [node]
            solution.append(route)
            continue

        place_in_route = random.randint(0, len(route))
        route.insert(place_in_route, node)

    return solution

def greedy_insertion(old_solution, nodes_to_insert, dists, length_info,
                     regret_num, args):
    """
    Applies the greedy insertion heuristic to a destroyed solution using
    the new nodes from the set covering heuristic.

    Parameters:
        old_solution: destroyed solution
        nodes_to_insert: set of nodes to insert
        dists: distance matrix
        instance_length_info: information about the route constraints
        regret_num: regret value to be applied
        args: command line arguments
    
    Outputs:
        solution: new solution
    """

    length_type, max_length = length_info
    solution = old_solution.copy()
    route_too_long = [False if len(r) < max_length else True for r in solution]

    while nodes_to_insert:
        c = np.zeros((len(nodes_to_insert), len(solution)))
        info =  [[] for _ in range(len(nodes_to_insert))]
        for i, node in enumerate(nodes_to_insert):
            for k, route in enumerate(solution):
                if length_type == 'vertices':
                    if len(route) >= max_length:
                        obj, place_in_route = np.inf, None
                        route_too_long[k] = True
                    else:
                        place = add_node_to_route(node, route, dists)
                        obj, place_in_route, route = place
                else:
                    obj, place_in_route, route = add_node_to_route(node, route, dists)
                    if obj >= max_length:
                        obj, place_in_route = np.inf, None
                        route_too_long[k] = True

                info[i].append((place_in_route, route))
                c[i, k] = obj

        if all(route_too_long) and nodes_to_insert != []:
            route_too_long.append(False)
            solution.append([nodes_to_insert.pop(0)])
            continue


        if all(flatten(np.isinf(c))):
            return solution, nodes_to_insert
        n, r = np.unravel_index(c.argmin(), c.shape)
        solution[r] = info[n][r][1]
        nodes_to_insert.pop(n)
    return solution

def add_node_to_route(node, route, dists):
    """
    Adds a node to a route in its best place.

    Parameters:
        node: node to insert
        route: route to insert node into
        dists: distance matrix
    
    Outputs:
        best_place: tuple containing the best place to insert the node and the
                    route with the node inserted
    """

    positions = len(route)+1
    objectives = []

    for i in range(positions):
        cur_route_nodes = route[:i] + [node] + route[i:]
        cur_route_edges = route_to_edges([0] + cur_route_nodes + [0])
        objectives.append((objective(cur_route_edges,
                                  dists, route=True), i, cur_route_nodes))
    
    return min(objectives, key=lambda x:x[0])
