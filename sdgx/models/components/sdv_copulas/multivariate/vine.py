"""VineCopula module."""

import logging
import sys
import warnings

import numpy as np
import pandas as pd

from sdgx.models.components.sdv_copulas import (
    EPSILON,
    check_valid_values,
    get_qualified_name,
    random_state,
    store_args,
    validate_random_state,
)
from sdgx.models.components.sdv_copulas.bivariate.base import Bivariate, CopulaTypes
from sdgx.models.components.sdv_copulas.multivariate.base import Multivariate
from sdgx.models.components.sdv_copulas.multivariate.tree import Tree, get_tree
from sdgx.models.components.sdv_copulas.univariate.gaussian_kde import GaussianKDE

LOGGER = logging.getLogger(__name__)


class VineCopula(Multivariate):
    """Vine copula model.

    A :math:`vine` is a graphical representation of one factorization of the n-variate probability
    distribution in terms of :math:`n(n − 1)/2` bivariate copulas by means of the chain rule.

    It consists of a sequence of levels and as many levels as variables. Each level consists of
    a tree (no isolated nodes and no loops) satisfying that if it has :math:`n` nodes there must
    be :math:`n − 1` edges.

    Each node in tree :math:`T_1` is a variable and edges are couplings of variables constructed
    with bivariate copulas.

    Each node in tree :math:`T_{k+1}` is a coupling in :math:`T_{k}`, expressed by the copula
    of the variables; while edges are couplings between two vertices that must have one variable
    in common, becoming a conditioning variable in the bivariate copula. Thus, every level has
    one node less than the former. Once all the trees are drawn, the factorization is the product
    of all the nodes.

    Args:
        vine_type (str):
            type of the vine copula, could be 'center','direct','regular'
        random_state (int or np.random.RandomState):
            Random seed or RandomState to use.


    Attributes:
        model (copulas.univariate.Univariate):
            Distribution to compute univariates.
        u_matrix (numpy.array):
            Univariates.
        n_sample (int):
            Number of samples.
        n_var (int):
            Number of variables.
        columns (pandas.Series):
            Names of the variables.
        tau_mat (numpy.array):
            Kendall correlation parameters for data.
        truncated (int):
            Max level used to build the vine.
        depth (int):
            Vine depth.
        trees (list[Tree]):
            List of trees used by this vine.
        ppfs (list[callable]):
            percent point functions from the univariates used by this vine.
    """

    @store_args
    def __init__(self, vine_type, random_state=None):
        if sys.version_info > (3, 8):
            warnings.warn(
                "Vines have not been fully tested on Python 3.8 and might "
                "produce wrong results. Please use Python 3.5, 3.6 or 3.7"
            )

        self.random_state = validate_random_state(random_state)
        self.vine_type = vine_type
        self.u_matrix = None

        self.model = GaussianKDE

    @classmethod
    def _deserialize_trees(cls, tree_list):
        previous = Tree.from_dict(tree_list[0])
        trees = [previous]

        for tree_dict in tree_list[1:]:
            tree = Tree.from_dict(tree_dict, previous)
            trees.append(tree)
            previous = tree

        return trees

    def to_dict(self):
        """Return a `dict` with the parameters to replicate this Vine.

        Returns:
            dict:
                Parameters of this Vine.
        """
        result = {
            "type": get_qualified_name(self),
            "vine_type": self.vine_type,
            "fitted": self.fitted,
        }

        if not self.fitted:
            return result

        result.update(
            {
                "n_sample": self.n_sample,
                "n_var": self.n_var,
                "depth": self.depth,
                "truncated": self.truncated,
                "trees": [tree.to_dict() for tree in self.trees],
                "tau_mat": self.tau_mat.tolist(),
                "u_matrix": self.u_matrix.tolist(),
                "unis": [distribution.to_dict() for distribution in self.unis],
                "columns": self.columns,
            }
        )
        return result

    @classmethod
    def from_dict(cls, vine_dict):
        """Create a new instance from a parameters dictionary.

        Args:
            params (dict):
                Parameters of the Vine, in the same format as the one
                returned by the ``to_dict`` method.

        Returns:
            Vine:
                Instance of the Vine defined on the parameters.
        """
        instance = cls(vine_dict["vine_type"])
        fitted = vine_dict["fitted"]
        if fitted:
            instance.fitted = fitted
            instance.n_sample = vine_dict["n_sample"]
            instance.n_var = vine_dict["n_var"]
            instance.truncated = vine_dict["truncated"]
            instance.depth = vine_dict["depth"]
            instance.trees = cls._deserialize_trees(vine_dict["trees"])
            instance.unis = [GaussianKDE.from_dict(uni) for uni in vine_dict["unis"]]
            instance.ppfs = [uni.percent_point for uni in instance.unis]
            instance.columns = vine_dict["columns"]
            instance.tau_mat = np.array(vine_dict["tau_mat"])
            instance.u_matrix = np.array(vine_dict["u_matrix"])

        return instance

    @check_valid_values
    def fit(self, X, truncated=3):
        """Fit a vine model to the data.

        1. Transform all the variables by means of their marginals.
        In other words, compute

        .. math:: u_i = F_i(x_i), i = 1, ..., n

        and compose the matrix :math:`u = u_1, ..., u_n,` where :math:`u_i` are their columns.

        Args:
            X (numpy.ndarray):
                Data to be fitted to.
            truncated (int):
                Max level to build the vine.
        """
        LOGGER.info('Fitting VineCopula("%s")', self.vine_type)
        self.n_sample, self.n_var = X.shape
        self.columns = X.columns
        self.tau_mat = X.corr(method="kendall").to_numpy()
        self.u_matrix = np.empty([self.n_sample, self.n_var])

        self.truncated = truncated
        self.depth = self.n_var - 1
        self.trees = []

        self.unis, self.ppfs = [], []
        for i, col in enumerate(X):
            uni = self.model()
            uni.fit(X[col])
            self.u_matrix[:, i] = uni.cumulative_distribution(X[col])
            self.unis.append(uni)
            self.ppfs.append(uni.percent_point)

        self.train_vine(self.vine_type)
        self.fitted = True

    def train_vine(self, tree_type):
        r"""Build the vine.

        1. For the construction of the first tree :math:`T_1`, assign one node to each variable
           and then couple them by maximizing the measure of association considered.
           Different vines impose different constraints on this construction. When those are
           applied different trees are achieved at this level.

        2. Select the copula that best fits to the pair of variables coupled by each edge in
           :math:`T_1`.

        3. Let :math:`C_{ij}(u_i , u_j )` be the copula for a given edge :math:`(u_i, u_j)`
           in :math:`T_1`. Then for every edge in :math:`T_1`, compute either

           .. math:: {v^1}_{j|i} = \\frac{\\partial C_{ij}(u_i, u_j)}{\\partial u_j}

           or similarly :math:`{v^1}_{i|j}`, which are conditional cdfs. When finished with
           all the edges, construct the new matrix with :math:`v^1` that has one less column u.

        4. Set k = 2.

        5. Assign one node of :math:`T_k` to each edge of :math:`T_ {k−1}`. The structure of
           :math:`T_{k−1}` imposes a set of constraints on which edges of :math:`T_k` are
           realizable. Hence the next step is to get a linked list of the accesible nodes for
           every node in :math:`T_k`.

        6. As in step 1, nodes of :math:`T_k` are coupled maximizing the measure of association
           considered and satisfying the constraints impose by the kind of vine employed plus the
           set of constraints imposed by tree :math:`T_{k−1}`.

        7. Select the copula that best fit to each edge created in :math:`T_k`.

        8. Recompute matrix :math:`v_k` as in step 4, but taking :math:`T_k` and :math:`vk−1`
           instead of :math:`T_1` and u.

        9. Set :math:`k = k + 1` and repeat from (5) until all the trees are constructed.

        Args:
            tree_type (str or TreeTypes):
                Type of trees to use.
        """
        LOGGER.debug("start building tree : 0")
        # 1
        tree_1 = get_tree(tree_type)
        tree_1.fit(0, self.n_var, self.tau_mat, self.u_matrix)
        self.trees.append(tree_1)
        LOGGER.debug("finish building tree : 0")

        for k in range(1, min(self.n_var - 1, self.truncated)):
            # get constraints from previous tree
            self.trees[k - 1]._get_constraints()
            tau = self.trees[k - 1].get_tau_matrix()
            LOGGER.debug(f"start building tree: {k}")
            tree_k = get_tree(tree_type)
            tree_k.fit(k, self.n_var - k, tau, self.trees[k - 1])
            self.trees.append(tree_k)
            LOGGER.debug(f"finish building tree: {k}")

    def get_likelihood(self, uni_matrix):
        """Compute likelihood of the vine."""
        num_tree = len(self.trees)
        values = np.empty([1, num_tree])

        for i in range(num_tree):
            value, new_uni_matrix = self.trees[i].get_likelihood(uni_matrix)
            uni_matrix = new_uni_matrix
            values[0, i] = value

        return np.sum(values)

    def _sample_row(self):
        """Generate a single sampled row from vine model.

        Returns:
            numpy.ndarray
        """
        unis = np.random.uniform(0, 1, self.n_var)
        # randomly select a node to start with
        first_ind = np.random.randint(0, self.n_var)
        adj = self.trees[0].get_adjacent_matrix()
        visited = []
        explore = [first_ind]

        sampled = np.zeros(self.n_var)
        itr = 0
        while explore:
            current = explore.pop(0)
            adj_is_one = adj[current, :] == 1
            neighbors = np.where(adj_is_one)[0].tolist()
            if itr == 0:
                new_x = self.ppfs[current](unis[current])

            else:
                for i in range(itr - 1, -1, -1):
                    current_ind = -1

                    if i >= self.truncated:
                        continue

                    current_tree = self.trees[i].edges
                    # get index of edge to retrieve
                    for edge in current_tree:
                        if i == 0:
                            if (edge.L == current and edge.R == visited[0]) or (
                                edge.R == current and edge.L == visited[0]
                            ):
                                current_ind = edge.index
                                break
                        else:
                            if edge.L == current or edge.R == current:
                                condition = set(edge.D)
                                condition.add(edge.L)  # noqa: PD005
                                condition.add(edge.R)  # noqa: PD005

                                visit_set = set(visited)
                                visit_set.add(current)  # noqa: PD005

                                if condition.issubset(visit_set):
                                    current_ind = edge.index
                                break

                    if current_ind != -1:
                        # the node is not indepedent contional on visited node
                        copula_type = current_tree[current_ind].name
                        copula = Bivariate(copula_type=CopulaTypes(copula_type))
                        copula.theta = current_tree[current_ind].theta

                        U = np.array([unis[visited[0]]])
                        if i == itr - 1:
                            tmp = copula.percent_point(np.array([unis[current]]), U)[0]
                        else:
                            tmp = copula.percent_point(np.array([tmp]), U)[0]

                        tmp = min(max(tmp, EPSILON), 0.99)

                new_x = self.ppfs[current](np.array([tmp]))

            sampled[current] = new_x

            for s in neighbors:
                if s not in visited:
                    explore.insert(0, s)

            itr += 1
            visited.insert(0, current)

        return sampled

    @random_state
    def sample(self, num_rows):
        """Sample new rows.

        Args:
            num_rows (int):
                Number of rows to sample

        Returns:
            pandas.DataFrame:
                sampled rows.
        """
        sampled_values = []
        for i in range(num_rows):
            sampled_values.append(self._sample_row())

        return pd.DataFrame(sampled_values, columns=self.columns)
