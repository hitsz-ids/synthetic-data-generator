"""Multivariate trees module."""

import logging
from enum import Enum

import numpy as np
import scipy

from sdgx.models.components.sdv_copulas import EPSILON, get_qualified_name
from sdgx.models.components.sdv_copulas.bivariate.base import Bivariate
from sdgx.models.components.sdv_copulas.multivariate.base import Multivariate

LOGGER = logging.getLogger(__name__)


class TreeTypes(Enum):
    """The available types of trees."""

    CENTER = 0
    DIRECT = 1
    REGULAR = 2


class Tree(Multivariate):
    """Helper class to instantiate a single tree in the vine model."""

    tree_type = None
    fitted = False

    def fit(self, index, n_nodes, tau_matrix, previous_tree, edges=None):
        """Fit this tree object.

        Args:
            index (int):
                index of the tree.
            n_nodes (int):
                number of nodes in the tree.
            tau_matrix (numpy.array):
                kendall's tau matrix of the data, shape (n_nodes, n_nodes).
            previous_tree (Tree):
                tree object of previous level.
        """
        self.level = index + 1
        self.n_nodes = n_nodes
        self.tau_matrix = tau_matrix
        self.previous_tree = previous_tree
        self.edges = edges or []

        if not self.edges:
            if self.level == 1:
                self.u_matrix = previous_tree
                self._build_first_tree()

            else:
                self._build_kth_tree()

            self.prepare_next_tree()

        self.fitted = True

    def _check_constraint(self, edge1, edge2):
        """Check if two edges satisfy vine constraint.

        Args:
            edge1 (Edge):
                edge object representing edge1
            edge2 (Edge):
                edge object representing edge2

        Returns:
            bool:
                True if the two edges satisfy vine constraints
        """
        full_node = {edge1.L, edge1.R, edge2.L, edge2.R}
        full_node.update(edge1.D)
        full_node.update(edge2.D)
        return len(full_node) == (self.level + 1)

    def _get_constraints(self):
        """Get neighboring edges for each edge in the edges."""
        num_edges = len(self.edges)
        for k in range(num_edges):
            for i in range(num_edges):
                # add to constraints if i shared an edge with k
                if k != i and self.edges[k].is_adjacent(self.edges[i]):
                    self.edges[k].neighbors.append(i)

    def _sort_tau_by_y(self, y):
        """Sort tau matrix by dependece with variable y.

        Args:
            y (int):
                index of variable of intrest

        Returns:
            numpy.ndarray:
                sorted tau matrix.
        """
        # first column is the variable of interest
        tau_y = self.tau_matrix[:, y]
        tau_y[y] = np.NaN

        temp = np.empty([self.n_nodes, 3])
        temp[:, 0] = np.arange(self.n_nodes)
        temp[:, 1] = tau_y
        temp[:, 2] = abs(tau_y)
        temp[np.isnan(temp)] = -10
        sort_temp = temp[:, 2].argsort()[::-1]
        tau_sorted = temp[sort_temp]

        return tau_sorted

    def get_tau_matrix(self):
        """Get tau matrix for adjacent pairs.

        Returns:
            tau (numpy.ndarray):
                tau matrix for the current tree
        """
        num_edges = len(self.edges)
        tau = np.empty([num_edges, num_edges])

        for i in range(num_edges):
            edge = self.edges[i]
            for j in edge.neighbors:
                if self.level == 1:
                    left_u = self.u_matrix[:, edge.L]
                    right_u = self.u_matrix[:, edge.R]

                else:
                    left_parent, right_parent = edge.parents
                    left_u, right_u = Edge.get_conditional_uni(left_parent, right_parent)

                tau[i, j], pvalue = scipy.stats.kendalltau(left_u, right_u)

        return tau

    def get_adjacent_matrix(self):
        """Get adjacency matrix.

        Returns:
            numpy.ndarray:
                adjacency matrix
        """
        edges = self.edges
        num_edges = len(edges) + 1
        adj = np.zeros([num_edges, num_edges])

        for k in range(num_edges - 1):
            adj[edges[k].L, edges[k].R] = 1
            adj[edges[k].R, edges[k].L] = 1

        return adj

    def prepare_next_tree(self):
        """Prepare conditional U matrix for next tree."""
        for edge in self.edges:
            copula_theta = edge.theta

            if self.level == 1:
                left_u = self.u_matrix[:, edge.L]
                right_u = self.u_matrix[:, edge.R]

            else:
                left_parent, right_parent = edge.parents
                left_u, right_u = Edge.get_conditional_uni(left_parent, right_parent)

            # compute conditional cdfs C(i|j) = dC(i,j)/duj and dC(i,j)/du
            left_u = [x for x in left_u if x is not None]
            right_u = [x for x in right_u if x is not None]
            X_left_right = np.array([[x, y] for x, y in zip(left_u, right_u)])
            X_right_left = np.array([[x, y] for x, y in zip(right_u, left_u)])

            copula = Bivariate(copula_type=edge.name)
            copula.theta = copula_theta
            left_given_right = copula.partial_derivative(X_left_right)
            right_given_left = copula.partial_derivative(X_right_left)

            # correction of 0 or 1
            left_given_right[left_given_right == 0] = EPSILON
            right_given_left[right_given_left == 0] = EPSILON
            left_given_right[left_given_right == 1] = 1 - EPSILON
            right_given_left[right_given_left == 1] = 1 - EPSILON
            edge.U = np.array([left_given_right, right_given_left])

    def get_likelihood(self, uni_matrix):
        """Compute likelihood of the tree given an U matrix.

        Args:
            uni_matrix (numpy.array):
                univariate matrix to evaluate likelihood on.

        Returns:
            tuple[float, numpy.array]:
                likelihood of the current tree, next level conditional univariate matrix
        """
        uni_dim = uni_matrix.shape[1]
        num_edge = len(self.edges)
        values = np.zeros([1, num_edge])
        new_uni_matrix = np.empty([uni_dim, uni_dim])

        for i in range(num_edge):
            edge = self.edges[i]
            value, left_u, right_u = edge.get_likelihood(uni_matrix)
            new_uni_matrix[edge.L, edge.R] = left_u
            new_uni_matrix[edge.R, edge.L] = right_u
            values[0, i] = np.log(value)

        return np.sum(values), new_uni_matrix

    def __str__(self):
        """Produce printable representation of the class."""
        template = "L:{} R:{} D:{} Copula:{} Theta:{}"
        return "\n".join(
            [template.format(edge.L, edge.R, edge.D, edge.name, edge.theta) for edge in self.edges]
        )

    def _serialize_previous_tree(self):
        if self.level == 1:
            return self.previous_tree.tolist()

        return None

    @classmethod
    def _deserialize_previous_tree(cls, tree_dict, previous):
        if tree_dict["level"] == 1:
            return np.array(tree_dict["previous_tree"])

        return previous

    def to_dict(self):
        """Return a `dict` with the parameters to replicate this Tree.

        Returns:
            dict:
                Parameters of this Tree.
        """
        fitted = self.fitted
        result = {"tree_type": self.tree_type, "type": get_qualified_name(self), "fitted": fitted}

        if not fitted:
            return result

        result.update(
            {
                "level": self.level,
                "n_nodes": self.n_nodes,
                "tau_matrix": self.tau_matrix.tolist(),
                "previous_tree": self._serialize_previous_tree(),
                "edges": [edge.to_dict() for edge in self.edges],
            }
        )

        return result

    @classmethod
    def from_dict(cls, tree_dict, previous=None):
        """Create a new instance from a parameters dictionary.

        Args:
            params (dict):
                Parameters of the Tree, in the same format as the one
                returned by the ``to_dict`` method.

        Returns:
            Tree:
                Instance of the tree defined on the parameters.
        """
        instance = get_tree(tree_dict["tree_type"])

        fitted = tree_dict["fitted"]
        instance.fitted = fitted
        if fitted:
            instance.level = tree_dict["level"]
            instance.n_nodes = tree_dict["n_nodes"]
            instance.tau_matrix = np.array(tree_dict["tau_matrix"])
            instance.previous_tree = cls._deserialize_previous_tree(tree_dict, previous)
            instance.edges = [Edge.from_dict(edge) for edge in tree_dict["edges"]]

        return instance


class CenterTree(Tree):
    """Tree for a C-vine copula."""

    tree_type = TreeTypes.CENTER

    def _build_first_tree(self):
        """Build first level tree."""
        tau_sorted = self._sort_tau_by_y(0)
        for itr in range(self.n_nodes - 1):
            ind = int(tau_sorted[itr, 0])
            copula = Bivariate.select_copula(self.u_matrix[:, (0, ind)])
            name, theta = copula.copula_type, copula.theta

            new_edge = Edge(itr, 0, ind, name, theta)
            new_edge.tau = self.tau_matrix[0, ind]
            self.edges.append(new_edge)

    def _build_kth_tree(self):
        """Build k-th level tree."""
        anchor = self.get_anchor()
        aux_sorted = self._sort_tau_by_y(anchor)
        edges = self.previous_tree.edges

        for itr in range(self.n_nodes - 1):
            right = int(aux_sorted[itr, 0])
            left_parent, right_parent = Edge.sort_edge([edges[anchor], edges[right]])
            new_edge = Edge.get_child_edge(itr, left_parent, right_parent)
            new_edge.tau = aux_sorted[itr, 1]
            self.edges.append(new_edge)

    def get_anchor(self):
        """Find anchor variable with highest sum of dependence with the rest.

        Returns:
            int:
                Anchor variable.
        """
        temp = np.empty([self.n_nodes, 2])
        temp[:, 0] = np.arange(self.n_nodes, dtype=int)
        temp[:, 1] = np.sum(abs(self.tau_matrix), 1)
        anchor = int(temp[0, 0])
        return anchor


class DirectTree(Tree):
    """DirectTree class."""

    tree_type = TreeTypes.DIRECT

    def _build_first_tree(self):
        # find the pair of maximum tau
        tau_matrix = self.tau_matrix
        tau_sorted = self._sort_tau_by_y(0)
        left_ind = tau_sorted[0, 0]
        right_ind = tau_sorted[1, 0]
        T1 = np.array([left_ind, 0, right_ind]).astype(int)
        tau_T1 = tau_sorted[:2, 1]

        # replace tau matrix of the selected variables as a negative number
        tau_matrix[:, [T1]] = -10
        for k in range(2, self.n_nodes - 1):
            left = np.argmax(tau_matrix[T1[0], :])
            right = np.argmax(tau_matrix[T1[-1], :])
            valL = np.max(tau_matrix[T1[0], :])
            valR = np.max(tau_matrix[T1[-1], :])

            if valL > valR:
                # add nodes to the left
                T1 = np.append(int(left), T1)
                tau_T1 = np.append(valL, tau_T1)
                tau_matrix[:, left] = -10

            else:
                # add node to the right
                T1 = np.append(T1, int(right))
                tau_T1 = np.append(tau_T1, valR)
                tau_matrix[:, right] = -10

        for k in range(self.n_nodes - 1):
            copula = Bivariate.select_copula(self.u_matrix[:, (T1[k], T1[k + 1])])
            name, theta = copula.copula_type, copula.theta

            left, right = sorted([T1[k], T1[k + 1]])
            new_edge = Edge(k, left, right, name, theta)
            new_edge.tau = tau_T1[k]
            self.edges.append(new_edge)

    def _build_kth_tree(self):
        edges = self.previous_tree.edges
        for k in range(self.n_nodes - 1):
            left_parent, right_parent = Edge.sort_edge([edges[k], edges[k + 1]])
            new_edge = Edge.get_child_edge(k, left_parent, right_parent)
            new_edge.tau = self.tau_matrix[k, k + 1]
            self.edges.append(new_edge)


class RegularTree(Tree):
    """RegularTree class."""

    tree_type = TreeTypes.REGULAR

    def _build_first_tree(self):
        """Build the first tree with n-1 variable."""
        # Prim's algorithm
        neg_tau = -1.0 * abs(self.tau_matrix)
        X = {0}

        while len(X) != self.n_nodes:
            adj_set = set()
            for x in X:
                for k in range(self.n_nodes):
                    if k not in X and k != x:
                        adj_set.add((x, k))  # noqa: PD005

            # find edge with maximum
            edge = sorted(adj_set, key=lambda e: neg_tau[e[0]][e[1]])[0]
            copula = Bivariate.select_copula(self.u_matrix[:, (edge[0], edge[1])])
            name, theta = copula.copula_type, copula.theta

            left, right = sorted([edge[0], edge[1]])
            new_edge = Edge(len(X) - 1, left, right, name, theta)
            new_edge.tau = self.tau_matrix[edge[0], edge[1]]
            self.edges.append(new_edge)
            X.add(edge[1])  # noqa: PD005

    def _build_kth_tree(self):
        """Build tree for level k."""
        neg_tau = -1.0 * abs(self.tau_matrix)
        edges = self.previous_tree.edges
        visited = {0}
        unvisited = set(range(self.n_nodes))

        while len(visited) != self.n_nodes:
            adj_set = set()
            for x in visited:
                for k in range(self.n_nodes):
                    # check if (x,k) is a valid edge in the vine
                    if k not in visited and k != x and self._check_constraint(edges[x], edges[k]):
                        adj_set.add((x, k))  # noqa: PD005

            # find edge with maximum tau
            if len(adj_set) == 0:
                visited.add(list(unvisited)[0])  # noqa: PD005
                continue

            pairs = sorted(adj_set, key=lambda e: neg_tau[e[0]][e[1]])[0]
            left_parent, right_parent = Edge.sort_edge([edges[pairs[0]], edges[pairs[1]]])

            new_edge = Edge.get_child_edge(len(visited) - 1, left_parent, right_parent)
            new_edge.tau = self.tau_matrix[pairs[0], pairs[1]]
            self.edges.append(new_edge)

            visited.add(pairs[1])  # noqa: PD005
            unvisited.remove(pairs[1])


def get_tree(tree_type):
    """Get a Tree instance of the specified type.

    Args:
        tree_type (str or TreeTypes):
            Type of tree of which to get an instance.

    Returns:
        Tree:
            Instance of a Tree of the specified type.
    """
    if not isinstance(tree_type, TreeTypes):
        if isinstance(tree_type, str) and tree_type.upper() in TreeTypes.__members__:
            tree_type = TreeTypes[tree_type.upper()]
        else:
            raise ValueError(f"Invalid tree type {tree_type}")

    if tree_type == TreeTypes.CENTER:
        return CenterTree()
    if tree_type == TreeTypes.REGULAR:
        return RegularTree()
    if tree_type == TreeTypes.DIRECT:
        return DirectTree()


class Edge(object):
    """Represents an edge in the copula."""

    def __init__(self, index, left, right, copula_name, copula_theta):
        """Initialize an Edge object.

        Args:
            left (int):
                left_node index (smaller)
            right (int):
                right_node index (larger)
            copula_name (str):
                name of the fitted copula class
            copula_theta (float):
                parameters of the fitted copula class
        """
        self.index = index
        self.L = left
        self.R = right
        self.D = set()  # dependence_set
        self.parents = None
        self.neighbors = []

        self.name = copula_name
        self.theta = copula_theta
        self.tau = None
        self.U = None
        self.likelihood = None

    @staticmethod
    def _identify_eds_ing(first, second):
        """Find nodes connecting adjacent edges.

        Args:
            first (Edge):
                Edge object representing the first edge.
            second (Edge):
                Edge object representing the second edge.

        Returns:
            tuple[int, int, set[int]]:
                The first two values represent left and right node
                indicies of the new edge. The third value is the new dependence set.
        """
        A = {first.L, first.R}
        A.update(first.D)

        B = {second.L, second.R}
        B.update(second.D)

        depend_set = A & B
        left, right = sorted(A ^ B)

        return left, right, depend_set

    def is_adjacent(self, another_edge):
        """Check if two edges are adjacent.

        Args:
            another_edge (Edge):
                edge object of another edge

        Returns:
            bool:
                True if the two edges are adjacent.
        """
        return (
            self.L == another_edge.L
            or self.L == another_edge.R
            or self.R == another_edge.L
            or self.R == another_edge.R
        )

    @staticmethod
    def sort_edge(edges):
        """Sort iterable of edges first by left node indices then right.

        Args:
            edges (list[Edge]):
                List of edges to be sorted.

        Returns:
            list[Edge]:
                Sorted list by left and right node indices.
        """
        return sorted(edges, key=lambda x: (x.L, x.R))

    @classmethod
    def get_conditional_uni(cls, left_parent, right_parent):
        """Identify pair univariate value from parents.

        Args:
            left_parent (Edge):
                left parent
            right_parent (Edge):
                right parent

        Returns:
            tuple[np.ndarray, np.ndarray]:
                left and right parents univariate.
        """
        left, right, _ = cls._identify_eds_ing(left_parent, right_parent)

        left_u = left_parent.U[0] if left_parent.L == left else left_parent.U[1]
        right_u = right_parent.U[0] if right_parent.L == right else right_parent.U[1]

        return left_u, right_u

    @classmethod
    def get_child_edge(cls, index, left_parent, right_parent):
        """Construct a child edge from two parent edges.

        Args:
            index (int):
                Index of the new Edge.
            left_parent (Edge):
                Left parent
            right_parent (Edge):
                Right parent

        Returns:
            Edge:
                The new child edge.
        """
        [ed1, ed2, depend_set] = cls._identify_eds_ing(left_parent, right_parent)
        left_u, right_u = cls.get_conditional_uni(left_parent, right_parent)
        X = np.array([[x, y] for x, y in zip(left_u, right_u)])
        copula = Bivariate.select_copula(X)
        name, theta = copula.copula_type, copula.theta
        new_edge = Edge(index, ed1, ed2, name, theta)
        new_edge.D = depend_set
        new_edge.parents = [left_parent, right_parent]
        return new_edge

    def get_likelihood(self, uni_matrix):
        """Compute likelihood given a U matrix.

        Args:
            uni_matrix (numpy.array):
                Matrix to compute the likelihood.

        Return:
            tuple (np.ndarray, np.ndarray, np.array):
                likelihood and conditional values.
        """
        if self.parents is None:
            left_u = uni_matrix[:, self.L]
            right_u = uni_matrix[:, self.R]

        else:
            left_ing = list(self.D - self.parents[0].D)[0]
            right_ing = list(self.D - self.parents[1].D)[0]
            left_u = uni_matrix[self.L, left_ing]
            right_u = uni_matrix[self.R, right_ing]

        copula = Bivariate(copula_type=self.name)
        copula.theta = self.theta

        X_left_right = np.array([[left_u, right_u]])
        X_right_left = np.array([[right_u, left_u]])

        value = np.sum(copula.probability_density(X_left_right))
        left_given_right = copula.partial_derivative(X_left_right)
        right_given_left = copula.partial_derivative(X_right_left)

        return value, left_given_right, right_given_left

    def to_dict(self):
        """Return a `dict` with the parameters to replicate this Edge.

        Returns:
            dict:
                Parameters of this Edge.
        """
        parents = None
        if self.parents:
            parents = [parent.to_dict() for parent in self.parents]

        U = None
        if self.U is not None:
            U = self.U.tolist()

        return {
            "index": self.index,
            "L": self.L,
            "R": self.R,
            "D": self.D,
            "parents": parents,
            "neighbors": self.neighbors,
            "name": self.name,
            "theta": self.theta,
            "tau": self.tau,
            "U": U,
            "likelihood": self.likelihood,
        }

    @classmethod
    def from_dict(cls, edge_dict):
        """Create a new instance from a parameters dictionary.

        Args:
            params (dict):
                Parameters of the Edge, in the same format as the one
                returned by the ``to_dict`` method.

        Returns:
            Edge:
                Instance of the edge defined on the parameters.
        """
        instance = cls(
            edge_dict["index"],
            edge_dict["L"],
            edge_dict["R"],
            edge_dict["name"],
            edge_dict["theta"],
        )
        instance.U = np.array(edge_dict["U"])
        parents = edge_dict["parents"]

        if parents:
            instance.parents = []
            for parent in parents:
                edge = Edge.from_dict(parent)
                instance.parents.append(edge)

        regular_attributes = ["D", "tau", "likelihood", "neighbors"]
        for key in regular_attributes:
            setattr(instance, key, edge_dict[key])

        return instance
