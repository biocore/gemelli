# ----------------------------------------------------------------------------
# Copyright (c) 2019--, gemelli development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import biom
import skbio
import numpy as np
import pandas as pd
from typing import Union, Optional
from skbio import TreeNode, OrdinationResults, DistanceMatrix
from gemelli.matrix_completion import MatrixCompletion
from gemelli.preprocessing import (matrix_rclr,
                                   fast_unifrac,
                                   bp_read_phylogeny,
                                   retrieve_t2t_taxonomy,
                                   create_taxonomy_metadata)
from gemelli._defaults import (DEFAULT_COMP, DEFAULT_MTD,
                               DEFAULT_MSC, DEFAULT_MFC,
                               DEFAULT_OPTSPACE_ITERATIONS,
                               DEFAULT_MFF)
from scipy.linalg import svd
# import QIIME2 if in a Q2env otherwise set type to str
try:
    from q2_types.tree import NewickFormat
except ImportError:
    # python does not check but technically this is the type
    NewickFormat = str


def phylogenetic_rpca_without_taxonomy(
        table: biom.Table,
        phylogeny: NewickFormat,
        n_components: Union[int, str] = DEFAULT_COMP,
        min_sample_count: int = DEFAULT_MSC,
        min_feature_count: int = DEFAULT_MFC,
        min_feature_frequency: float = DEFAULT_MFF,
        min_depth: int = DEFAULT_MTD,
        max_iterations: int = DEFAULT_OPTSPACE_ITERATIONS) -> (
        OrdinationResults, DistanceMatrix,
        TreeNode, biom.Table):
    """
    Runs phylogenetic RPCA without taxonomy. This code will
    be run QIIME2 versions of gemelli. Outside of QIIME2
    please use phylogenetic_rpca.
    """

    output = phylogenetic_rpca(table=table,
                               phylogeny=phylogeny,
                               n_components=n_components,
                               min_sample_count=min_sample_count,
                               min_feature_count=min_feature_count,
                               min_feature_frequency=min_feature_frequency,
                               min_depth=min_depth,
                               max_iterations=max_iterations)
    ord_res, dist_res, phylogeny, counts_by_node, _ = output

    return ord_res, dist_res, phylogeny, counts_by_node


def phylogenetic_rpca_with_taxonomy(
            table: biom.Table,
            phylogeny: NewickFormat,
            taxonomy: pd.DataFrame,
            n_components: Union[int, str] = DEFAULT_COMP,
            min_sample_count: int = DEFAULT_MSC,
            min_feature_count: int = DEFAULT_MFC,
            min_feature_frequency: float = DEFAULT_MFF,
            min_depth: int = DEFAULT_MTD,
            max_iterations: int = DEFAULT_OPTSPACE_ITERATIONS) -> (
        OrdinationResults, DistanceMatrix,
        TreeNode, biom.Table, pd.DataFrame):
    """
    Runs phylogenetic RPCA with taxonomy. This code will
    be run QIIME2 versions of gemelli. Outside of QIIME2
    please use phylogenetic_rpca.
    """
    output = phylogenetic_rpca(table=table,
                               phylogeny=phylogeny,
                               taxonomy=taxonomy,
                               n_components=n_components,
                               min_sample_count=min_sample_count,
                               min_feature_count=min_feature_count,
                               min_feature_frequency=min_feature_frequency,
                               min_depth=min_depth,
                               max_iterations=max_iterations)
    ord_res, dist_res, phylogeny, counts_by_node, result_taxonomy = output

    return ord_res, dist_res, phylogeny, counts_by_node, result_taxonomy


def phylogenetic_rpca(table: biom.Table,
                      phylogeny: NewickFormat,
                      taxonomy: Optional[pd.DataFrame] = None,
                      n_components: Union[int, str] = DEFAULT_COMP,
                      min_sample_count: int = DEFAULT_MSC,
                      min_feature_count: int = DEFAULT_MFC,
                      min_feature_frequency: float = DEFAULT_MFF,
                      min_depth: int = DEFAULT_MTD,
                      max_iterations: int = DEFAULT_OPTSPACE_ITERATIONS) -> (
                          OrdinationResults, DistanceMatrix,
                          TreeNode, biom.Table, Optional[pd.DataFrame]):
    """
    Performs robust phylogenetic center log-ratio transform and
    robust PCA. The robust PCA and enter log-ratio transform
    operate on only observed values of the data.
    For more information see (1 and 2).

    Parameters
    ----------
    table: numpy.ndarray, required
    The feature table in biom format containing the
    samples over which metric should be computed.

    phylogeny: str, required
    Path to the file containing the phylogenetic tree containing tip
    identifiers that correspond to the feature identifiers in the table.
    This tree can contain tip ids that are not present in the table,
    but all feature ids in the table must be present in this tree.

    taxonomy: pd.DataFrame, optional
    Taxonomy file in QIIME2 formatting. See the feature metdata
    section of https://docs.qiime2.org/2021.11/tutorials/metadata

    n_components: int, optional : Default is 3
    The underlying rank of the data and number of
    output dimentions.

    min_sample_count: int, optional : Default is 0
    Minimum sum cutoff of sample across all features.
    The value can be at minimum zero and must be an
    whole integer. It is suggested to be greater than
    or equal to 500.

    min_feature_count: int, optional : Default is 0
    Minimum sum cutoff of features across all samples.
    The value can be at minimum zero and must be
    an whole integer.

    min_feature_frequency: float, optional : Default is 0
    Minimum percentage of samples a feature must appear
    with a value greater than zero. This value can range
    from 0 to 100 with decimal values allowed.

    max_iterations: int, optional : Default is 5
    The number of convex iterations to optimize the solution
    If iteration is not specified, then the default iteration is 5.
    Which redcues to a satisfactory error threshold.

    Returns
    -------
    OrdinationResults
        A biplot of the (Robust Aitchison) RPCA feature loadings

    DistanceMatrix
        The Aitchison distance of the sample loadings from RPCA.

    TreeNode
        The input tree with all nodes matched in name to the
        features in the counts-by-node table

    biom.Table
        A table with all tree internal nodes as features with the
        sum of all children of that node (i.e. FastUniFrac).

    Optional[pd.DataFrame]
        The resulting tax2Tree taxonomy and will include taxonomy for both
        internal nodes and tips. Note: this will only be output
        if taxonomy was given as input.

    Raises
    ------
    ValueError
        `ValueError: n_components must be at least 2`.

    ValueError
        `ValueError: max_iterations must be at least 1`.

    ValueError
        `ValueError: Data-table contains either np.inf or -np.inf`.

    ValueError
        `ValueError: The n_components must be less
            than the minimum shape of the input table`.

    References
    ----------
    .. [1] Martino C, Morton JT, Marotz CA, Thompson LR, Tripathi A,
           Knight R, Zengler K. 2019. A Novel Sparse Compositional
           Technique Reveals Microbial Perturbations. mSystems 4.
    .. [2] Keshavan RH, Oh S, Montanari A. 2009. Matrix completion
            from a few entries (2009_ IEEE International
            Symposium on Information Theory

    Examples
    --------
    import numpy as np
    import pandas as pd
    from biom import Table
    from gemelli.rpca import phylogenetic_rpca

    # make a table
    X = np.array([[9, 3, 0, 0],
                [9, 9, 0, 1],
                [0, 1, 4, 5],
                [0, 0, 3, 4],
                [1, 0, 8, 9]])
    sample_ids = ['s1','s2','s3','s4']
    feature_ids = ['f1','f2','f3','f4','f5']
    bt = Table(X, feature_ids, sample_ids)
    # write an example tree to read
    f = open("demo-tree.nwk", "w")
    newick = '(((f1:1,f2:1)n9:1,f3:1)n8:1,(f4:1,f5:1)n2:1)n1:1;'
    f.write(newick)
    f.close()
    # run RPCA without taxonomy
    # s1/s2 will seperate from s3/s4
    (ordination, distance_matrix,
    tree, phylo_table, _) = phylogenetic_rpca(bt, 'demo-tree.nwk')

    # make mock taxonomy
    taxonomy = pd.DataFrame({fid:['k__kingdom; p__phylum;'
                                'c__class; o__order; '
                                'f__family; g__genus;'
                                's__',
                                0.99]
                            for fid in feature_ids},
                            ['Taxon', 'Confidence']).T
    # run RPCA with taxonomy
    # s1/s2 will seperate from s3/s4
    (ordination, distance_matrix,
    tree, phylo_table,
    lca_taxonomy) = phylogenetic_rpca(bt, 'demo-tree.nwk', taxonomy)

    """

    # use helper to process table
    table = rpca_table_processing(table,
                                  min_sample_count,
                                  min_feature_count,
                                  min_feature_frequency)

    # import the tree based on filtered table
    phylogeny = bp_read_phylogeny(table,
                                  phylogeny,
                                  min_depth)
    # build the vectorized table
    counts_by_node, tree_index, branch_lengths, fids, otu_ids\
        = fast_unifrac(table, phylogeny)
    # Robust-clt (matrix_rclr) preprocessing
    rclr_table = matrix_rclr(counts_by_node, branch_lengths=branch_lengths)
    # run OptSpace (RPCA)
    ord_res, dist_res = optspace_helper(rclr_table, fids, table.ids(),
                                        n_components=n_components)
    # import expanded table
    counts_by_node = biom.Table(counts_by_node.T,
                                fids, table.ids())
    result_taxonomy = None
    if taxonomy is not None:
        # collect taxonomic information for all tree nodes.
        traversed_taxonomy = retrieve_t2t_taxonomy(phylogeny, taxonomy)
        result_taxonomy = create_taxonomy_metadata(phylogeny,
                                                   traversed_taxonomy)

    return ord_res, dist_res, phylogeny, counts_by_node, result_taxonomy


def rpca(table: biom.Table,
         n_components: Union[int, str] = DEFAULT_COMP,
         min_sample_count: int = DEFAULT_MSC,
         min_feature_count: int = DEFAULT_MFC,
         min_feature_frequency: float = DEFAULT_MFF,
         max_iterations: int = DEFAULT_OPTSPACE_ITERATIONS) -> (
        OrdinationResults,
        DistanceMatrix):
    """
    Performs robust center log-ratio transform and
    robust PCA. The robust PCA and enter log-ratio transform
    operate on only observed values of the data.
    For more information see (1 and 2).

    Parameters
    ----------
    table: numpy.ndarray, required
    The feature table in biom format containing the
    samples over which metric should be computed.

    n_components: int, optional : Default is 3
    The underlying rank of the data and number of
    output dimentions.

    min_sample_count: int, optional : Default is 0
    Minimum sum cutoff of sample across all features.
    The value can be at minimum zero and must be an
    whole integer. It is suggested to be greater than
    or equal to 500.

    min_feature_count: int, optional : Default is 0
    Minimum sum cutoff of features across all samples.
    The value can be at minimum zero and must be
    an whole integer.

    min_feature_frequency: float, optional : Default is 0
    Minimum percentage of samples a feature must appear
    with a value greater than zero. This value can range
    from 0 to 100 with decimal values allowed.

    max_iterations: int, optional : Default is 5
    The number of convex iterations to optimize the solution
    If iteration is not specified, then the default iteration is 5.
    Which redcues to a satisfactory error threshold.

    Returns
    -------
    OrdinationResults
        A biplot of the (Robust Aitchison) RPCA feature loadings

    DistanceMatrix
        The Aitchison distance of the sample loadings from RPCA.

    Raises
    ------
    ValueError
        `ValueError: n_components must be at least 2`.

    ValueError
        `ValueError: max_iterations must be at least 1`.

    ValueError
        `ValueError: Data-table contains either np.inf or -np.inf`.

    ValueError
        `ValueError: The n_components must be less
            than the minimum shape of the input table`.

    References
    ----------
    .. [1] Martino C, Morton JT, Marotz CA, Thompson LR, Tripathi A,
           Knight R, Zengler K. 2019. A Novel Sparse Compositional
           Technique Reveals Microbial Perturbations. mSystems 4.
    .. [2] Keshavan RH, Oh S, Montanari A. 2009. Matrix completion
            from a few entries (2009_ IEEE International
            Symposium on Information Theory

    Examples
    --------
    import numpy as np
    from biom import Table
    from gemelli.rpca import rpca

    # make a table
    X = np.array([[9, 3, 0, 0],
                [9, 9, 0, 1],
                [0, 1, 4, 5],
                [0, 0, 3, 4],
                [1, 0, 8, 9]])
    sample_ids = ['s1','s2','s3','s4']
    feature_ids = ['f1','f2','f3','f4','f5']
    bt = Table(X, feature_ids, sample_ids)
    # run RPCA (s1/s2 will seperate from s3/s4)
    ordination, distance_matrix = rpca(bt)

    """
    # use helper to process table
    table = rpca_table_processing(table,
                                  min_sample_count,
                                  min_feature_count,
                                  min_feature_frequency)
    # Robust-clt (matrix_rclr) preprocessing
    rclr_table = matrix_rclr(table.matrix_data.toarray().T)
    # run OptSpace (RPCA)
    ord_res, dist_res = optspace_helper(rclr_table,
                                        table.ids('observation'),
                                        table.ids(), n_components=n_components)

    return ord_res, dist_res


def optspace_helper(rclr_table: np.array,
                    feature_ids: list,
                    subject_ids: list,
                    n_components: Union[int, str] = DEFAULT_COMP,
                    max_iterations: int = DEFAULT_OPTSPACE_ITERATIONS) -> (
                        OrdinationResults,
                        DistanceMatrix):
    """
    Helper function. Please use rpca directly.
    """
    # run OptSpace (RPCA)
    opt = MatrixCompletion(n_components=n_components,
                           max_iterations=max_iterations).fit(rclr_table)
    # get new n-comp when applicable
    n_components = opt.s.shape[0]
    # get PC column labels for the skbio OrdinationResults
    rename_cols = ['PC' + str(i + 1) for i in range(n_components)]
    # get completed matrix for centering
    X = opt.sample_weights @ opt.s @ opt.feature_weights.T
    # center again around zero after completion
    X = X - X.mean(axis=0)
    X = X - X.mean(axis=1).reshape(-1, 1)
    # re-factor the data
    u, s, v = svd(X)
    # only take n-components
    u = u[:, :n_components]
    v = v.T[:, :n_components]
    # calc. the new variance using projection
    p = s**2 / np.sum(s**2)
    p = p[:n_components]
    s = s[:n_components]
    # save the loadings
    feature_loading = pd.DataFrame(v, index=feature_ids,
                                   columns=rename_cols)
    sample_loading = pd.DataFrame(u, index=subject_ids,
                                  columns=rename_cols)
    # % var explained
    proportion_explained = pd.Series(p, index=rename_cols)
    # get eigenvalues
    eigvals = pd.Series(s, index=rename_cols)

    # if the n_components is two add PC3 of zeros
    # this is referenced as in issue in
    # <https://github.com/biocore/emperor/commit
    # /a93f029548c421cb0ba365b4294f7a5a6b0209ce>
    # discussed in gemelli -- PR#29
    if n_components == 2:
        feature_loading['PC3'] = [0] * len(feature_loading.index)
        sample_loading['PC3'] = [0] * len(sample_loading.index)
        eigvals.loc['PC3'] = 0
        proportion_explained.loc['PC3'] = 0

    # save ordination results
    short_method_name = 'rpca_biplot'
    long_method_name = '(Robust Aitchison) RPCA Biplot'
    ord_res = skbio.OrdinationResults(
        short_method_name,
        long_method_name,
        eigvals.copy(),
        samples=sample_loading.copy(),
        features=feature_loading.copy(),
        proportion_explained=proportion_explained.copy())
    # save distance matrix
    dist_res = DistanceMatrix(opt.distance, ids=sample_loading.index)

    return ord_res, dist_res


def auto_rpca(table: biom.Table,
              min_sample_count: int = DEFAULT_MSC,
              min_feature_count: int = DEFAULT_MFC,
              min_feature_frequency: float = DEFAULT_MFF,
              max_iterations: int = DEFAULT_OPTSPACE_ITERATIONS) -> (
        OrdinationResults,
        DistanceMatrix):
    """Runs RPCA but with auto estimation of the
       rank peramater.
    """
    ord_res, dist_res = rpca(table,
                             n_components='auto',
                             min_sample_count=min_sample_count,
                             min_feature_count=min_feature_count,
                             min_feature_frequency=min_feature_frequency,
                             max_iterations=max_iterations)
    return ord_res, dist_res


def rpca_table_processing(table: biom.Table,
                          min_sample_count: int = DEFAULT_MSC,
                          min_feature_count: int = DEFAULT_MFC,
                          min_feature_frequency: float = DEFAULT_MFF) -> (
                              biom.Table):
    """Filter and checks the table validity for RPCA.
    """
    # get shape of table
    n_features, n_samples = table.shape

    # filter sample to min seq. depth
    def sample_filter(val, id_, md):
        return sum(val) > min_sample_count

    # filter features to min total counts
    def observation_filter(val, id_, md):
        return sum(val) > min_feature_count

    # filter features by N samples presence
    def frequency_filter(val, id_, md):
        return (np.sum(val > 0) / n_samples) > (min_feature_frequency / 100)

    # filter and import table for each filter above
    table = table.filter(observation_filter, axis='observation')
    table = table.filter(frequency_filter, axis='observation')
    table = table.filter(sample_filter, axis='sample')

    # check the table after filtering
    if len(table.ids()) != len(set(table.ids())):
        raise ValueError('Data-table contains duplicate sample IDs')
    if len(table.ids('observation')) != len(set(table.ids('observation'))):
        raise ValueError('Data-table contains duplicate feature IDs')

    return table
