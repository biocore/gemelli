# ----------------------------------------------------------------------------
# Copyright (c) 2019--, gemelli development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import warnings
import io
import sys
import os
import t2t.nlevel as nl
import numpy as np
import pandas as pd
from biom import Table
from skbio import TreeNode
from .base import _BaseConstruct
from gemelli._defaults import DEFAULT_MTD
from skbio.diversity._util import _vectorize_counts_and_tree
from bp import parse_newick, to_skbio_treenode
# import QIIME2 if in a Q2env otherwise set type to str
try:
    from q2_types.tree import NewickFormat
except ImportError:
    # python does not check but technically this is the type
    NewickFormat = str


VALID_TAXONOMY_COLUMN_NAMES = ('taxon', 'taxonomy')


class TaxonomyError(Exception):
    pass


def create_taxonomy_metadata(phylogeny, traversed_taxonomy=None):
    """
        Create a pd.DataFrame with index 'Feature ID' and column 'Taxon'.

        This method will traverse phylogeny using TreeNode.traverse() and
        use the node names found in phylogeny as the index 'Feature ID'.
        Thus, traversed_taxonomy must be parallel List of phylogeny (i.e. a
        List that layes out the taxonomy in a traverse() fashion).

        Parameters
        ----------
        phylogeny: TreeNode
        traversed_taxonomy: List
    """
    f_id = []
    tax = []
    if traversed_taxonomy is not None:
        # add taxonomy for all nodes in tree
        for i, node in enumerate(phylogeny.traverse(include_self=True)):
            # metadata.append([node.name, traversed_taxonomy[i]])
            f_id.append(node.name)
            tax.append(traversed_taxonomy[i])
    else:
        # create empty dataframe
        f_id = ['None']
        tax = ['None']
    returned_taxonomy = pd.DataFrame(data={'Feature ID': f_id, 'Taxon': tax})
    returned_taxonomy.set_index(keys='Feature ID', inplace=True)
    return returned_taxonomy


def retrieve_t2t_taxonomy(phylogeny, taxonomy=None):
    """
    Returns a List containing the taxonomy of all nodes in the tree stored in
    TreeNode.traverse() order. If taxonomy is None, then None will be returned

    based on :
    https://github.com/biocore/tax2tree/blob/9b3814fb19e935c06a31e61e848d0f91bcecb305/scripts/t2t#L46

    Parameters
    ----------
    phylogeny: TreeNode
    taxonomy : pd.DataFrame with Index 'Feature ID' and contains a column
        'taxon' or 'taxonomy' (case insensitive)
    """
    if taxonomy is None:
        # return empty taxonomy
        return None

    # make copy of phylogeny
    consensus_tree = phylogeny.copy()
    # validate and convert taxonomy into a StringIO stream
    consensus_map = _get_taxonomy_io_stream(taxonomy)

    tipname_map = nl.load_consensus_map(consensus_map, False)
    tree_ = nl.load_tree(consensus_tree, tipname_map)
    counts = nl.collect_names_at_ranks_counts(tree_)

    nl.decorate_ntips(tree_)
    nl.decorate_name_relative_freqs(tree_, counts, 2)
    nl.set_ranksafe(tree_)
    nl.pick_names(tree_)

    nl.set_preliminary_name_and_rank(tree_)

    contree, contree_lookup = nl.make_consensus_tree(tipname_map.values())
    # Disable print statements
    with open(os.devnull, 'w') as fpnull:
        sys.stdout = fpnull
    sys.stdout = open(os.devnull, 'w')
    nl.backfill_names_gap(tree_, contree_lookup)
    # Restore print statements
    sys.stdout = sys.__stdout__
    nl.commonname_promotion(tree_)

    constrings = _pull_consensus_strings(tree_)
    return constrings


def _pull_consensus_strings(consensus_tree):
    """
    Pulls consensus strings off of consensus_tree. Assumes .name is set
    This is a helper function for retrieve_t2t_taxonomy.

    based on:
    https://github.com/biocore/tax2tree/blob/9b3814fb19e935c06a31e61e848d0f91bcecb305/t2t/nlevel.py#L831

    Parameters
    ----------
    consensus_tree: TreeNode
    """
    constrings = []
    rank_order = {r: i for i, r in enumerate(nl.RANK_ORDER)}

    # helper function
    def _add_name_to_consensus_string(node_name, cons_string):
        # only add if node has a name
        if node_name:
            if ';' in node_name:
                names = [r.strip() for r in node_name.split(';')]
                for node_name in names:
                    rank_idx = rank_order[node_name[0]]
                    cons_string[rank_idx] = node_name
            else:
                rank_idx = rank_order[node_name[0]]
                cons_string[rank_idx] = node_name

    # start at each node and travel up
    for node in consensus_tree.traverse(include_self=True):
        consensus_string = ['%s__' % r for r in nl.RANK_ORDER]
        # internal nodes will have taxonomy for name so we need to add it to
        # the consensus_string
        if not node.is_tip():
            _add_name_to_consensus_string(node.name, consensus_string)
        if node.is_root():
            constrings.append('; '.join(consensus_string))
            continue
        p = node.parent
        # walk up the consensus_tree filling in the consensus string
        while p.parent:
            _add_name_to_consensus_string(p.name, consensus_string)
            p = p.parent

        # if there is a name at the root we need to make sure we grab it
        if p.name:
            _add_name_to_consensus_string(p.name, consensus_string)

        # join strings with tip id
        constrings.append('; '.join(consensus_string))
    return constrings


def _get_taxonomy_io_stream(taxonomy):
    """
    Returns a StringIO of taxonomy.
    This is a hepler function for retrieve_t2t_taxonomy.
    Raises
    ------
    TaxonomyError
        taxonomy does not contain a valid taxonomy column (see
        VALID_TAXONOMY_COLUMN_NAMES )
    """
    lowercase_col_names = [str(c).lower() for c in taxonomy.columns]

    # See if there is a "taxonomy column", and do some related validation on
    # column names
    tax_col_index = None
    tax_col_name = None
    for i, col in enumerate(lowercase_col_names):
        if col in VALID_TAXONOMY_COLUMN_NAMES:
            if tax_col_index is None:
                tax_col_index = i
                # ("col" has already been set to lowercase)
                tax_col_name = taxonomy.columns[i]
            else:
                # Error condition 1 -- multiple possible "taxonomy columns" :(
                raise TaxonomyError(
                    (
                        "Multiple columns in the feature metadata have one of "
                        "the following names (case insensitive): {}. At most "
                        "one feature metadata column can have a name from "
                        "that list."
                    ).format(VALID_TAXONOMY_COLUMN_NAMES)
                )
    if tax_col_name is None:
        raise TaxonomyError(
            (
                "The taxonomy file does not contain a column with one the "
                "following (case insensitive): {}."
            ).format(VALID_TAXONOMY_COLUMN_NAMES)
        )
    stream = io.StringIO()

    # Split the single column of taxonomy strings into n columns, where n
    # is the highest number of taxonomic levels in any string. This is to
    # ensure that all feateures have the same amount of levels for t2t
    taxonomy = taxonomy[tax_col_name]\
        .str.strip().str.split(r'\s*;\s*', expand=True)

    #  set t2t rank order
    highest_rank_indx = taxonomy.dropna(axis=0, how='any').first_valid_index()
    rank_order = [rank[0]
                  for rank in taxonomy.loc[highest_rank_indx].values.tolist()]
    nl.set_rank_order(rank_order)

    # collapse taxonomy back into a single string
    taxonomy.fillna('', inplace=True)
    taxonomy = taxonomy.apply(
        lambda levels: '; '.join(levels.values.tolist()),
        axis=1)

    # convert taxonomy dataframe to a StringIO for use in t2t
    taxonomy.to_csv(stream, sep='\t', index=True, header=False)

    # set stream to point to first line
    stream.seek(0)

    return stream


def bp_read_phylogeny(table: Table,
                      phylogeny: NewickFormat,
                      min_depth: int = DEFAULT_MTD):
    """
    Fast way to read in phylogeny in newick
    format, filter, and return in TreeNode format.

    Parameters
    ----------
    table: biom.Table - a table of shape (M,N)
        N = Features (i.e. OTUs, metabolites)
        M = Samples
    phylogeny: str - path to file/data
                     in newick format
    min_depth: int
        Minimum number of total number of
        descendants (tips) to include a node.
        Default value of zero will retain all nodes
        (including tips).
    Examples
    --------
    TODO

    """

    # import file path
    with open(str(phylogeny)) as treefile:
        # read balanced parentheses tree
        phylogeny = parse_newick(treefile.readline())
        # first filter out
        names_to_keep = set((table.ids('observation')).flatten())
        phylogeny = phylogeny.shear(names_to_keep).collapse()
        # convert the tree to skbio TreeNode for processing
        phylogeny = to_skbio_treenode(phylogeny)
        # filter internal nodes based on topology
        tree_topology_filter(phylogeny, min_depth=min_depth)

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
                                     phylogeny: NewickFormat,
                                     min_depth: int = DEFAULT_MTD) -> (
                                         Table, Table, TreeNode):
    """
    Takes biom table and returns fast_unifrac style
    vectorized count table and a matrix_rclr
    transformed biom table.

    """

    # import the tree and filter
    phylogeny = bp_read_phylogeny(table,
                                  phylogeny,
                                  min_depth)
    # build the vectorized table
    counts_by_node, tree_index, branch_lengths, fids, otu_ids\
        = fast_unifrac(table, phylogeny)
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


def fast_unifrac(table, tree):
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
    # combine filters
    keep_node = (keep_zero & node_branch_zero)
    # subset the table (if need, otherwise ignored)
    counts_by_node = counts_by_node[:, keep_node]
    branch_lengths = branch_lengths[keep_node]
    fids = ['n' + i for i in list(tree_index['id'][keep_node].astype(str))]
    tree_index['keep'] = {i: v for i, v in enumerate(keep_node)}
    # re-label tree to return with labels
    tree_relabel = {tid_: tree_index['id_index'][int(tid_[1:])]
                    for tid_ in fids}
    # re-name nodes to match vectorized table
    otu_ids_set = set(otu_ids)
    for new_id, node_ in tree_relabel.items():
        if node_.name in otu_ids_set:
            # replace table name (leaf - nondup)
            fids[fids.index(new_id)] = node_.name
        else:
            # replace tree name (internal)
            node_.name = new_id

    return counts_by_node, tree_index, branch_lengths, fids, otu_ids


def tree_topology_filter(tree, min_depth=DEFAULT_MTD):
    """
    A tree topology filter based on the
    number of descendants. This function
    only removes internal nodes. Tips are
    moved to the parent of the removed node.

    In part, original function comes from
    https://github.com/biocore/wol
    kindly provided here by Qiyun Zhu.
    ----------
    tree : skbio.TreeNode
        tree to calculate metrics
    min_depth: int
        Minimum number of total number of
        descendants (tips) to include a node.
        Default value of zero will retain all nodes.
    Notes
    -----
    The following metrics will be calculated for each node:
    - n : int
        number of descendants (tips)
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
    >>> tree_topology_filter(tree, min_depth=2)
    >>> print(tree.ascii_art())
                               /-C
                              |
                     /n8------|--A
                    |         |
                    |          \\-B
           /n4------|
          |         |--D
          |         |
          |          \\-E
          |
          |            /-F
          |            |
    -n1------|         |--G
             |-n3------|
             |         |--H
             |         |
             |         \\-I
             |
             |--J
             |
             \\-K


    """

    # calculate bottom-up metrics
    for node in tree.postorder(include_self=True):
        if node.is_tip():
            node.n = 1
        else:
            children = node.children
            node.n = sum(x.n for x in children)
    # check to ensure tree filters make sense
    # (this has to be done after building the metrics above)
    if tree.root().n <= min_depth:
        raise ValueError('min_depth is equal to tree root value, '
                         'this will result in a table of zero '
                         'features.')
    # non-tip nodes to remove (below the depth filter)
    nodes_to_remove = [node for node in tree.postorder(include_self=True)
                       if (not node.is_tip()) & (node.n <= min_depth)]
    # remove nodes in the tree by moving tips up to parents
    for node_to_remove in nodes_to_remove:
        tips_to_move_up = list(node_to_remove.tips())
        for tip_to_move_up in tips_to_move_up:
            node_to_remove.parent.append(tip_to_move_up)
        node_to_remove.parent.remove(node_to_remove)
    # reconstruct correct topology after removing nodes
    tree.prune()


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

    def construct(self, table, mf, subjects, conditions, branch_lengths=None):
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
        self.feature_order = np.lexsort((table.sum(1).index,
                                         -table.sum(1).values))
        self.feature_order = table.index[self.feature_order]
        self.table = self.table.loc[self.feature_order, :]
        self.mf = mf.copy()
        self.subjects = subjects
        self.conditions = conditions
        self.branch_lengths = branch_lengths
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
        rlcr_table = matrix_rclr(table_counts.T,
                                 branch_lengths=self.branch_lengths).T

        # Step 2: fill the tensor (missing are all zero)

        # sort by data within groups to avoid
        # different results with permuted labels
        mf['seq_depth'] = table.sum(0).reindex(mf.index).fillna(0)

        def sortset(grp_col, mf_tmp):
            order_tmp = mf_tmp.groupby(grp_col).sum()
            order_tmp = order_tmp.sort_values('seq_depth').index
            return list(order_tmp)

        # get the ordered subjects
        subject_order = sortset(self.subjects, mf)
        # get un-ordered features (order does not matter)
        feature_order = list(table.index)
        # get the ordered for each conditional
        conditional_orders = [sortset(cond, mf)
                              for cond in self.conditions]
        # generate the dims.
        all_dim = [subject_order]\
            + conditional_orders  # concat all

        # get tensor to fill with counts (all zeros)
        shape = tuple([len(cl) for cl in [subject_order,
                                          feature_order]
                       + all_dim[1:]])
        tensor_counts = np.zeros(tuple(shape))
        tensor_rclr = tensor_counts.copy()

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
            tensor_rclr[tuple(ind_)] = rlcr_table[:, M_ind]
        tensor_rclr[~np.isfinite(tensor_rclr)] = 0.0

        # save metadat and save subject-conditional index
        condition_metadata_map = [{(sid, con): i
                                  for i, sid, con in zip(mf.index,
                                                         mf[self.subjects],
                                                         mf[con])}
                                  for con in self.conditions]
        self.condition_metadata_map = condition_metadata_map
        # save tensor label order
        self.counts = tensor_counts
        self.rclr_transformed = tensor_rclr
        self.subject_order = subject_order
        self.feature_order = feature_order
        self.condition_orders = conditional_orders
        self.mf = self.mf
