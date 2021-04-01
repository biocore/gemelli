# ----------------------------------------------------------------------------
# Copyright (c) 2019--, gemelli development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import warnings
import numpy as np
from biom import Table
from skbio import TreeNode
from .base import _BaseConstruct
from gemelli._defaults import DEFAULT_MTD
from skbio.diversity._util import _vectorize_counts_and_tree
from bp import parse_newick, to_skbio_treenode


def bp_read_phylogeny(table, phylogeny):
    """
    Fast way to read in phylogeny in newick
    format and return in TreeNode format.

    Parameters
    ----------
    table: biom.Table - a table of shape (M,N)
        N = Features (i.e. OTUs, metabolites)
        M = Samples

    phylogeny: str - path to file/data
                     in newick format

    Examples
    --------
    TODO

    """

    # import file path
    with open(str(phylogeny)) as treefile:
        # The file will still be closed even though we return from within the
        # with block: see https://stackoverflow.com/a/9885287/10730311.
        phylogeny = parse_newick(treefile.readline())
        phylogeny = phylogeny.shear(set((table.ids('observation')).flatten()))
        phylogeny = to_skbio_treenode(phylogeny)

    return phylogeny


def tensor_rclr(T, branch_lengths=None):
    """
    Robust clr transform. is the approximate geometric mean of X.

    We know from the Central Limit Theorem that as
    we collect more independent measurements we
    approach the true geometric mean.

    This transformation will work on N mode tensors
    by flattening. Flattened tensor are reshaped into
    subject x contions by features before transformation.
    A tensor will be returned in the shape shape and order.

    Mode 2 tensors (matrix) will be directly transformed,
    no reshaping necessary.

    Parameters
    ----------
    T : array-like
        Array of non-negative count data.
        In an N mode tensor of shape:
        first dimension = samples
        second dimension = features
        [3..N] dimensions = conditions

    Raises
    ------
    ValueError
        Tensor is less than 2-dimensions.
    ValueError
        Tensor contains negative values.
    ValueError
        Tensor contains np.inf or -np.inf.
    ValueError
        Tensor contains np.nan or missing.

    References
    ----------
    .. [1] V. Pawlowsky-Glahn, J. J. Egozcue,
           R. Tolosana-Delgado (2015),
           Modeling and Analysis of
           Compositional Data, Wiley,
           Chichester, UK

    .. [2] C. Martino et al., A Novel Sparse
           Compositional Technique Reveals
           Microbial Perturbations. mSystems.
           4 (2019), doi:10.1128/mSystems.00016-19.

    Examples
    --------
    TODO

    """

    if len(T.shape) < 2:
        raise ValueError('Tensor is less than 2-dimensions')

    if np.count_nonzero(np.isinf(T)) != 0:
        raise ValueError('Tensor contains either np.inf or -np.inf.')

    if np.count_nonzero(np.isnan(T)) != 0:
        raise ValueError('Tensor contains np.nan or missing.')

    if (T < 0).any():
        raise ValueError('Tensor contains negative values.')

    if len(T.shape) < 3:
        # tensor_rclr on 2D matrix
        M_tensor_rclr = matrix_rclr(T.transpose().copy(),
                                    branch_lengths=branch_lengths).T
        M_tensor_rclr[~np.isfinite(M_tensor_rclr)] = 0.0
        return M_tensor_rclr
    else:
        # flatten tensor (samples*conditions x features)
        T = T.copy()
        # conditional dimensions
        conditions_index = list(range(2, len(T.shape)))
        forward_T = tuple([0] + conditions_index + [1])
        reverse_T = tuple([0] + [conditions_index[-1]]
                          + [1] + conditions_index[:-1])
        # transpose to flatten
        T = T.transpose(forward_T)
        M = T.reshape(np.product(T.shape[:len(T.shape) - 1]),
                      T.shape[-1])
        with np.errstate(divide='ignore', invalid='ignore'):
            M_tensor_rclr = matrix_rclr(M, branch_lengths=branch_lengths)
        M_tensor_rclr[~np.isfinite(M_tensor_rclr)] = 0.0
        # reshape to former tensor and return tensors
        return M_tensor_rclr.reshape(T.shape).transpose(reverse_T)


def matrix_rclr(mat, branch_lengths=None):
    """
    Robust clr transform helper function.
    This function is built for mode 2 tensors,
    also known as matrices.

    Raises
    ------
    ValueError
       Raises an error if any values are negative.
    ValueError
       Raises an error if the matrix has more than 2 dimension.

    References
    ----------
    .. [1] V. Pawlowsky-Glahn, J. J. Egozcue,
           R. Tolosana-Delgado (2015),
           Modeling and Analysis of
           Compositional Data, Wiley,
           Chichester, UK

    .. [2] C. Martino et al., A Novel Sparse
           Compositional Technique Reveals
           Microbial Perturbations. mSystems.
           4 (2019), doi:10.1128/mSystems.00016-19.

    Examples
    --------
    TODO

    """
    # ensure array is at least 2D
    mat = np.atleast_2d(np.array(mat))
    # ensure array not more than 2D
    if mat.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    # ensure no neg values
    if (mat < 0).any():
        raise ValueError('Array Contains Negative Values')
    # ensure no undefined values
    if np.count_nonzero(np.isinf(mat)) != 0:
        raise ValueError('Data-matrix contains either np.inf or -np.inf')
    # ensure no missing values
    if np.count_nonzero(np.isnan(mat)) != 0:
        raise ValueError('Data-matrix contains nans')
    # take the log of the sample centered data
    if branch_lengths is not None:
        mat = np.log(matrix_closure(matrix_closure(mat) * branch_lengths))
    else:
        mat = np.log(matrix_closure(mat))
    # generate a mask of missing values
    mask = [True] * mat.shape[0] * mat.shape[1]
    mask = np.array(mat).reshape(mat.shape)
    mask[np.isfinite(mat)] = False
    # sum of rows (features)
    lmat = np.ma.array(mat, mask=mask)
    # perfrom geometric mean
    gm = lmat.mean(axis=-1, keepdims=True)
    # center with the geometric mean
    lmat = (lmat - gm).squeeze().data
    # mask the missing with nan
    lmat[~np.isfinite(mat)] = np.nan
    return lmat


def rclr_transformation(table: Table) -> Table:
    """
    Takes biom table and returns
    a matrix_rclr transformed biom table.
    """
    # transform table values (and return biom.Table)
    table = Table(matrix_rclr(table.matrix_data.toarray().T).T,
                  table.ids('observation'),
                  table.ids('sample'))
    return table


def phylogenetic_rclr_transformation(table: Table,
                                     phylogeny: TreeNode,
                                     min_depth: int = DEFAULT_MTD,
                                     min_splits: int = DEFAULT_MTD,
                                     max_postlevel: int = DEFAULT_MTD) -> (
                                         Table, Table, TreeNode):
    """
    Takes biom table and returns fast_unifrac style
    vectorized count table and a matrix_rclr
    transformed biom table.

    """
    # build the vectorized table
    counts_by_node, tree_index, branch_lengths, fids, otu_ids\
        = fast_unifrac(table, phylogeny, min_depth, min_splits, max_postlevel)
    # Robust-clt (matrix_rclr) preprocessing
    rclr_table = matrix_rclr(counts_by_node, branch_lengths=branch_lengths)
    # import transformed matrix into biom.Table
    rclr_table = Table(rclr_table.T,
                       fids, table.ids('sample'))
    # import expanded matrix into biom.Table
    counts_by_node = Table(counts_by_node.T,
                           fids, table.ids())

    return counts_by_node, rclr_table, phylogeny


def matrix_closure(mat):
    """
    Simillar to the skbio.stats.composition.closure function.
    Performs closure to ensure that all elements add up to 1.
    However, this function allows for zero rows. This results
    in rows that may contain missing (NaN) vlues. These
    all zero rows may occur as a product of a tensor slice and
    is dealt later with the tensor restructuring and factorization.

    Parameters
    ----------
    mat : array_like
       a matrix of proportions where
       rows = compositions
       columns = components
    Returns
    -------
    array_like, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1
    Examples
    --------
    >>> import numpy as np
    >>> from gemelli.preprocessing import matrix_closure
    >>> X = np.array([[2, 2, 6], [0, 0, 0]])
    >>> closure(X)
    array([[ 0.2,  0.2,  0.6],
           [ nan,  nan,  nan]])

    """

    mat = np.atleast_2d(mat)
    mat = mat / mat.sum(axis=1, keepdims=True)

    return mat.squeeze()


def fast_unifrac(table, tree, min_depth=0, min_splits=0, max_postlevel=0):
    """
    A wrapper to return a vectorized Fast UniFrac
    algorithm. The nodes up the tree are summed
    and exposed as vectors in the matrix. The
    closed matrix is then multipled by the
    branch lengths to phylogenically
    weight the data.

    Parameters
    ----------
    table : biom.Table
       A biom table of counts.
    tree: skbio.TreeNode
       Tree containing the features in the table.
    min_depth: int
        Minimum number of total number of
        descendants (tips) to include a node.
        Default value of zero will retain all nodes
        (including tips).
    min_splits: int
        Minimum number of total number of
        splits to include a node.
        Default value of zero will retain all nodes
        (including tips).
    max_postlevel: int
        Minimum allowable max postlevel
        splits to include a node.
        Default value of zero will retain all nodes
        (including tips).
    Returns
    -------
    counts_by_node: array_like, np.float64
       A matrix of counts with internal nodes
       vectorized.
    tree_index: dict
        A housekeeping dictionary.
    branch_lengths: array_like, np.float64
        An array of branch lengths.
    fids: list
        A list of feature IDs matched to tree_index['id'].
    otu_ids: list
        A list of the original table OTU IDs (tips).
    Examples
    --------
    TODO

    """

    # original table
    bt_array = table.matrix_data.toarray()
    otu_ids = table.ids('observation')
    # expand the vectorized table
    counts_by_node, tree_index, branch_lengths \
        = _vectorize_counts_and_tree(bt_array.T, otu_ids, tree)
    # check branch lengths
    if sum(branch_lengths) == 0:
        raise ValueError('All tree branch lengths are zero. '
                         'This will result in a table of zero features.')
    # drop zero sum features (non-optional for CTF/RPCA)
    keep_zero = counts_by_node.sum(0) > 0
    # drop zero branch_lengths (no point to keep it)
    node_branch_zero = branch_lengths.sum(0) > 0
    # calculate node information
    calc_split_metrics(tree)
    # check to ensure tree filters make sense
    if tree.root().n <= min_depth:
        raise ValueError('min_depth is equal to tree root value, '
                         'this will result in a table of zero '
                         'features.')
    if np.max(tree.root().postlevels) <= max_postlevel:
        raise ValueError('max_postlevel is equal to max postlevel at tree'
                         ' root, this will result in a table of zero '
                         'features.')
    if tree.root().splits <= min_splits:
        raise ValueError('min_splits is equal to number of splits at tree'
                         ' root, this will result in a table of zero '
                         'features.')
    # create index to filter table
    keep_node_depth = np.array([True] * counts_by_node.shape[1])
    keep_node_splits = np.array([True] * counts_by_node.shape[1])
    keep_node_postlevel = np.array([True] * counts_by_node.shape[1])
    for count_index_, node_ in tree_index['id_index'].items():
        # total number nodes under is more than min_depth
        filter_tmp_ = node_.n > min_depth
        keep_node_depth[count_index_] = filter_tmp_
        # number of splits
        filter_tmp_ = node_.splits >= min_splits
        keep_node_splits[count_index_] = filter_tmp_
        # number of postlevel
        filter_tmp_ = np.max(node_.postlevels) > max_postlevel
        keep_node_postlevel[count_index_] = filter_tmp_
    # get joint set of nodes to keep
    keep_node = (keep_node_depth & keep_node_postlevel &
                 keep_node_splits & keep_zero & node_branch_zero)
    # check all filter
    if sum(keep_node) == 0:
        raise ValueError('Combined table and tree filters resulted'
                         ' in a table of zero features.')
    # subset the table
    counts_by_node = counts_by_node[:, keep_node]
    branch_lengths = branch_lengths[keep_node]
    fids = ['n' + i for i in list(tree_index['id'][keep_node].astype(str))]
    tree_index['keep'] = {i: v for i, v in enumerate(keep_node)}
    # re-label tree to return with labels
    tree_relabel = {tid_: tree_index['id_index'][int(tid_[1:])]
                    for tid_ in fids}
    # re-name nodes to match vectorized table
    for new_id, node_ in tree_relabel.items():
        if node_.name in otu_ids:
            # replace table name (leaf - nondup)
            fids[fids.index(new_id)] = node_.name
        else:
            # replace tree name (internal)
            node_.name = new_id

    return counts_by_node, tree_index, branch_lengths, fids, otu_ids


def calc_split_metrics(tree):
    """Calculate topology split-related metrics.
    Parameters. Original function comes from
    https://github.com/biocore/wol
    kindly provided here by Qiyun Zhu.
    ----------
    tree : skbio.TreeNode
        tree to calculate metrics
    Notes
    -----
    The following metrics will be calculated for each node:
    - n : int
        number of descendants (tips)
    - splits : int
        total number of splits from tips
    - prelevel : int
        number of nodes from root
    Examples
    --------
    >>> # Example from Fig. 9a of Puigbo, et al., 2009, J Biol:
    >>> newick = '((((A,B)n9,C)n8,(D,E)n7)n4,((F,G)n6,(H,I)n5)n3,(J,K)n2)n1;'
    >>> tree = TreeNode.read([newick])
    >>> print(tree.ascii_art())
                                            /-A
                                  /n9------|
                        /n8------|          \\-B
                       |         |
              /n4------|          \\-C
             |         |
             |         |          /-D
             |          \\n7------|
             |                    \\-E
             |
             |                    /-F
    -n1------|          /n6------|
             |         |          \\-G
             |-n3------|
             |         |          /-H
             |          \\n5------|
             |                    \\-I
             |
             |          /-J
              \\n2------|
                        \\-K
    >>> calc_split_metrics(tree)
    >>> tree.find('n3').n
    4
    >>> tree.find('n4').splits
    4
    >>> tree.find('n8').postlevels
    [3, 3, 2]

    """
    # calculate bottom-up metrics
    for node in tree.postorder(include_self=True):
        if node.is_tip():
            node.n = 1
            node.splits = 0
            node.postlevels = [1]
        else:
            children = node.children
            node.n = sum(x.n for x in children)
            node.splits = sum(x.splits for x in children) + 1
            node.postlevels = [y + 1 for x in node.children for y in
                               x.postlevels]


class build(_BaseConstruct):

    """
    This class can both build N-mode
    tensors from 2D dataframes given
    a count table and mapping data.

    A list of conditional measurements are
    given that identify context measured
    multiple times over the same subjects.
    Additionally a set of subject IDs
    must be provided. Any subjects that are
    missing in a given condition are left
    as completely zero.

    Parameters
    ----------
    table : DataFrame
        table of non-negative count data
        rows = features
        columns = samples
    mapping : DataFrame
        mapping metadata for table
        rows = samples
        columns = categories
    subjects : str, int, or float
        category of sample IDs in metadata
    conditions : str, int, or float
        category of conditional in metadata

    Attributes
    -------
    subject_order : list
        order of subjects in tensor array
    feature_order : list
        order of features in tensor array
    condition_orders : list of lists
        order of conditions for each
        condition in tensor array
    counts : array-like
        Contains table counts.
        N mode tensor of shape
        first dimension = samples
        second dimension = features
        [3..N] dimensions = conditions

    Raises
    ------
    ValueError
        if subject not in mapping
    ValueError
        if any conditions not in mapping
    ValueError
        Table is not 2-dimensional
    ValueError
        Table contains negative values
    ValueError
        Table contains np.inf or -np.inf
    ValueError
        Table contains np.nan or missing.
    Warning
        If a conditional-sample pair
        has multiple IDs associated
        with it the multiple samples
        are meaned.

    References
    ----------
    .. [1] V. Pawlowsky-Glahn, J. J. Egozcue, R. Tolosana-Delgado (2015),
    Modeling and Analysis of Compositional Data, Wiley, Chichester, UK

    .. [2] C. Martino et al., A Novel Sparse Compositional Technique Reveals
    Microbial Perturbations. mSystems. 4 (2019), doi:10.1128/mSystems.00016-19.

    Examples
    --------
    TODO

    """

    def __init__(self):
        """
        Parameters
        ----------
        None

        """

    def construct(self, table, mf, subjects, conditions):
        """
        This function transforms a 2D table
        into a N-Order tensor.

        Parameters
        ----------
        table : DataFrame
            table of non-negative count data
            rows = features
            columns = samples
        mapping : DataFrame
            mapping metadata for table
            rows = samples
            columns = categories
        subjects : str, int, or float
            category of sample IDs in metadata
        conditions : list of strings or ints
            category of conditional in metadata

        Returns
        -------
        self to abstract method

        Raises
        ------
        ValueError
            if subject not in mapping
        ValueError
            if any conditions not in mapping
        ValueError
            Table is not 2-dimensional
        ValueError
            Table contains negative values
        ValueError
            Table contains np.inf or -np.inf
        ValueError
            Table contains np.nan or missing.
        Warning
            If a conditional-sample pair
            has multiple IDs associated
            with it. In this case the
            default method is to mean them.

        Examples
        --------
        TODO

        """

        if subjects not in mf.columns:
            raise ValueError("Subject provided (" +
                             str(subjects) +
                             ") category not in metadata columns.")

        if any(cond_col not in mf.columns for cond_col in conditions):
            missin_cond = ','.join([cond_col for cond_col in conditions
                                    if cond_col not in mf.columns])
            raise ValueError("Conditional category(s) [" +
                             str(missin_cond) +
                             "] not in metadata column(s).")

        if np.count_nonzero(np.isinf(table.values)) != 0:
            raise ValueError('Table contains either np.inf or -np.inf.')

        if np.count_nonzero(np.isnan(table.values)) != 0:
            raise ValueError('Table contains np.nan or missing.')

        if (table.values < 0).any():
            raise ValueError('Table contains negative values.')

        # store all to self
        self.table = table.copy()
        self.mf = mf.copy()
        self.subjects = subjects
        self.conditions = conditions
        self._construct()

        return self

    def _construct(self):
        """
        This function forms a tensor
        with missing subject x condition
        pairs left as all zeros.

        Raises
        ------
        Warning
            If a conditional-subject pair
            has multiple samples associated
            with it. In this case the
            default method is to mean them.

        """

        table, mf = self.table, self.mf

        # Step 1: mean samples with multiple conditional overlaps
        col_tmp = [self.subjects] + self.conditions
        duplicated = {k: list(df.index)
                      for k, df in mf.groupby(col_tmp)
                      if df.shape[0] > 1}  # get duplicated conditionals
        if len(duplicated.keys()) > 0:
            duplicated_ids = ','.join(list(set([str(k[0])
                                                for k in duplicated.keys()])))
            warnings.warn(''.join(["Subject(s) (", str(duplicated_ids),
                                   ") contains multiple ",
                                   "samples. Multiple subject counts will be",
                                   " meaned across samples by subject."]),
                          RuntimeWarning)
        for id_, dup in duplicated.items():
            # mean and keep one
            table[dup[0]] = table.loc[:, dup].mean(axis=1).astype(int)
            # drop the other
            table.drop(dup[1:], axis=1)
            mf.drop(dup[1:], axis=0)
        # save direct data
        table_counts = table.values

        # Step 2: fill the tensor (missing are all zero)

        # generate all sorted mode ids
        def sortset(ids): return sorted(set(ids))
        # get the ordered subjects
        subject_order = sortset(mf[self.subjects])
        # get un-ordered features (order does not matter)
        feature_order = list(table.index)
        # get the ordered for each conditional
        conditional_orders = [sortset(mf[cond])
                              for cond in self.conditions]
        # generate the dims.
        all_dim = [subject_order]\
            + conditional_orders  # concat all

        # get tensor to fill with counts (all zeros)
        shape = tuple([len(cl) for cl in [subject_order,
                                          feature_order]
                       + all_dim[1:]])
        tensor_counts = np.zeros(tuple(shape))

        # generate map from ordered subject and conditions
        # to the original orders in the table
        projection = {
            tuple(
                dim.index(k_) for k_, dim in zip(
                    k, all_dim)): list(
                table.columns).index(
                    df.index[0]) for k, df in mf.groupby(
                        [
                            self.subjects] + self.conditions)}

        # fill the tensor with data
        for T_ind, M_ind in projection.items():
            # get the index from the tensor
            ind_ = [T_ind[:1], list(range(len(table.index)))] + list(T_ind[1:])
            # fill count tensor from table
            tensor_counts[tuple(ind_)] = table_counts[:, M_ind]

        # save metadat and save subject-conditional index
        condition_metadata_map = [{(sid, con): i
                                  for i, sid, con in zip(mf.index,
                                                         mf[self.subjects],
                                                         mf[con])}
                                  for con in self.conditions]
        self.condition_metadata_map = condition_metadata_map
        # save tensor label order
        self.counts = tensor_counts
        self.subject_order = subject_order
        self.feature_order = feature_order
        self.condition_orders = conditional_orders
        self.mf = self.mf
