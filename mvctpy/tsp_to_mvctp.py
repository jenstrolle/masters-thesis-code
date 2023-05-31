import numpy as np

class tsp_to_mvctp_instance():
    """
    Takes TSPLib instance and adapts to MVCTP instance under
    the following rules:

    - V and T are defined as the first T_size and V_size nodes respectively.
      Remaining nodes are assigned to W.
    - The distances c_ij are taken as the Euclidean distances between pairs of
      vertices -- with the larges covering distance c taken as the maximum of
      the maximal distance between some j and the node in V \\ T which is the
      second closest node to j and the maximum of c_hj, for all h in V \\ T and
      j in W. This leads to each node in W being covered by at least two
      vertices in V \\ T and each vertex in V \\ T covering at least one vertex
      in W.
    """

    def __init__(self, path):
        """
        Initializes tsp_to_mvctp_instance class.

        Parameters:
            path (str) : path to the .tsp file
        """

        self.nodes, self.length_info, self.V_size, self.T_size = self.open_file(path)
        self.n, _ = self.nodes.shape
        V_size = self.V_size
        T_size = self.T_size
        self.V = list(range(0, V_size))
        self.W = list(range(V_size, self.n))
        self.T = list(range(0, T_size))
        self.num_W = self.n - self.V_size
        self.distances = self.get_distance_matrix()

        # setting length_info according to jozefowiez
        if self.length_info[0] == 'route_lengths':
            self.length_info[1] = 2*np.max(self.distances[1:V_size, 0])+self.length_info[1]

        # 1 is the depot node
        self.W_distances = self.distances[V_size:, T_size:V_size]
        self.W_V_distances = self.distances[V_size:, :V_size]
        self.covering_distance = self.get_covering_distance()
        self.covering_matrix = self.get_covering_matrix()
        self.all_covering = np.where(self.distances <= self.covering_distance,
                                     1, 0)

    def out(self):
        """
        Returns the parameters of the MVCTP instance.

        Output:
            V (list): list of nodes in V

            W (list): list of nodes in W

            T (list): list of nodes in T

            distances (np.array): distance matrix

            all_covering (np.array): binary matrix with entries d_ij equal to 1 if and only
                            if c_ij <= c for all v_i in V and v_j in V \\ T

            covering_distance (float): maximal distance a node in W can be from a node
                                in V and still be covered
        """
        return self.V, self.W, self.T, self.distances, self.all_covering, self.covering_distance

    def open_file(self, path):
        """
        Returns np.array containing points from .tsp file. ONLY handles .tsp
        files.

        Parameters:
            path (str): Path to .tsp file.

        Outputs:
            array (np.array): Array of shape n x 2 containing Euclidean coordinates
            of points in problem instance.
        """
        with open(path, 'r') as fi:
            list_file = fi.read().splitlines()

        length_type_idx = list_file.index('LENGTH_TYPE')+1
        length_type = list_file[length_type_idx]
        max_length = float(list_file[length_type_idx+1])
        length_constraints =  [length_type, max_length]

        num_v_index = list_file.index('NUMBER_OF_V_NODES')+1
        num_v = int(list_file[num_v_index])

        num_t_index = list_file.index('NUMBER_OF_T_NODES')+1
        num_t = int(list_file[num_t_index])

        node_index = list_file.index('NODE_COORD_SECTION')+1
        node_list = list_file[node_index:-1]

        point_list = [tuple(map(int, p.split()[1:])) for p in node_list]

        return np.array(point_list), length_constraints, num_v, num_t

    def get_distance_matrix(self):
        """
        Returns distance matrix C for MVCTP problem instance.

        Parameters:
            self (self): containing nodes and n.

        Outputs:
            array (np.array): Symmetric np.array of shape n x n
                              containing pairwise distances between
                              points in problem instance.
        """
        n = self.n
        x = self.nodes

        result = np.empty((n, n))
        for i in range(n):
            result[:, i] = np.sqrt(np.sum(np.abs(x-x[i])**2, axis=-1))

        return result

    def get_covering_distance(self):
        """
        Returns float c, the maximal distance a node in W can be from a node
        in V and still be covered.

        Parameters:
            self (self): containing W_distances, n and V.
        """
        second_shortest_indices = np.argsort(self.W_distances)[:, 1]
        second_shortest_list = self.W_distances[np.arange(self.n-self.V_size),
                                                second_shortest_indices]

        maximal_second_shortest_dist = np.max(second_shortest_list)

        max_min_distances = np.max(np.min(self.W_distances, axis=1))

        return max(max_min_distances, maximal_second_shortest_dist)

    def get_covering_matrix(self, matrix=None):
        """
        Returns binary np.array of shape (n-V_size x V_size-T_size) with entries
        d_lh equal to 1 if and only if c_lh <= c for v_l in W and v_h in V \\ T.

        Parameters:
            self (self): containing covering_distance and W_distances
        """

        if matrix is None:
            matrix = self.W_distances

        covering_matrix = np.where(matrix <= self.covering_distance,
                                   1, 0)


        return covering_matrix
