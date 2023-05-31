from itertools import chain

def get_coords(pts):
    """
    Returns the x and y coordinates of a list of points.
    """
    return list(zip(*pts))

def route_to_edges(route):
    """
    Returns a list of edges from a route. 
    """
    edges = []
    for idx, _ in enumerate(route):
        if idx == len(route)-1:
            break
        edges.append((route[idx], route[idx+1]))

    return edges

def objective(edge_routes, distances, route=False):
    """
    Returns the objective value of a solution.
    """
    total = 0
    if route:
        for edge in edge_routes:
            total += distances[edge]
        return total

    else:
        for edges in edge_routes:
            for e in edges:
                total += distances[e]
        return total

def simple_nodes_to_actual_indices(solution, actual):
    """
    Used by Shaw Removal heuristic to convert the indices of the nodes in the
    solution to the actual indices of the nodes in the graph.
    """
    all_routes = []
    for route in solution:
        if isinstance(route, int):
            return [actual[i] for i in solution]
        all_routes.append([actual[i] for i in route])

    return all_routes

def routes_with_destruction_removed(destruction, solution):
    """
    Returns the routes in a solution with the nodes in the destruction removed.
    """
    return [[n for n in route if n not in destruction] for route in solution]

def flatten(list_of_lists):
    """
    Flattens a list of lists. 
    """
    return list(chain(*list_of_lists))
