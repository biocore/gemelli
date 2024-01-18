# ----------------------------------------------------------------------------
# Copyright (c) 2019--, gemelli development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import biom
import skbio
import warnings
import numpy as np
import pandas as pd
from scipy.spatial import distance
from typing import Union, Optional
from skbio import TreeNode, OrdinationResults, DistanceMatrix
from gemelli.matrix_completion import MatrixCompletion
from gemelli.optspace import OptSpace
from gemelli.preprocessing import (matrix_rclr,
                                   fast_unifrac,
                                   bp_read_phylogeny,
                                   retrieve_t2t_taxonomy,
                                   create_taxonomy_metadata,
                                   mask_value_only)
from gemelli._defaults import (DEFAULT_COMP, DEFAULT_MTD,
                               DEFAULT_MSC, DEFAULT_MFC,
                               DEFAULT_OPTSPACE_ITERATIONS,
                               DEFAULT_MFF, DEFAULT_METACV,
                               DEFAULT_COLCV, DEFAULT_TESTS,
                               DEFAULT_MATCH, DEFAULT_TRNSFRM)
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
    output dimensions.

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
    Which reduces to a satisfactory error threshold.

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

    # validate the metadata using q2 as a wrapper
    if taxonomy is not None and not isinstance(taxonomy,
                                               pd.DataFrame):
        taxonomy = taxonomy.to_dataframe()
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
    output dimensions.

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
    Which reduces to a satisfactory error threshold.

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


def rpca_with_cv(table: biom.Table,
                 n_test_samples: int = DEFAULT_TESTS,
                 sample_metadata: pd.DataFrame = DEFAULT_METACV,
                 train_test_column: str = DEFAULT_COLCV,
                 n_components: Union[int, str] = DEFAULT_COMP,
                 max_iterations: int = DEFAULT_OPTSPACE_ITERATIONS,
                 min_sample_count: int = DEFAULT_MSC,
                 min_feature_count: int = DEFAULT_MFC,
                 min_feature_frequency: float = DEFAULT_MFF) -> (
                 OrdinationResults,
                 DistanceMatrix,
                 pd.DataFrame):
    """
    RPCA but with CV used in Joint-RPCA.

    Parameters
    ----------
    table: numpy.ndarray, required
    The feature table in biom format containing the
    samples over which metric should be computed.

    n_test_samples: int, optional : Default is 10
    Number of random samples to choose for test split samples.

    metadata: DataFrame, optional : Default is None
    Sample metadata file in QIIME2 formatting. The file must
    contain a train-test column with labels `train` and `test`
    and the row ids matched to the table(s).

    train_test_column: str, optional : Default is None
    Sample metadata column containing `train` and `test`
    labels to use for the cross-validation evaluation.

    n_components: int, optional : Default is 3
    The underlying rank of the data and number of
    output dimensions.

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
    Which reduces to a satisfactory error threshold.

    Returns
    -------
    OrdinationResults
        A biplot of the (Robust Aitchison) RPCA feature loadings

    DistanceMatrix
        The Aitchison distance of the sample loadings from RPCA.

    DataFrame
        The cross-validation reconstruction error.

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

    """
    res_tmp = joint_rpca([table],
                         n_test_samples=n_test_samples,
                         sample_metadata=sample_metadata,
                         train_test_column=train_test_column,
                         n_components=n_components,
                         max_iterations=max_iterations,
                         min_sample_count=min_sample_count,
                         min_feature_count=min_feature_count,
                         min_feature_frequency=min_feature_frequency)
    ord_res, dist_res, cv_dist = res_tmp
    return ord_res, dist_res, cv_dist


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


def joint_rpca(tables: biom.Table,
               n_test_samples: int = DEFAULT_TESTS,
               sample_metadata: pd.DataFrame = DEFAULT_METACV,
               train_test_column: str = DEFAULT_COLCV,
               n_components: Union[int, str] = DEFAULT_COMP,
               rclr_transform_tables: bool = DEFAULT_TRNSFRM,
               min_sample_count: int = DEFAULT_MSC,
               min_feature_count: int = DEFAULT_MFC,
               min_feature_frequency: float = DEFAULT_MFF,
               max_iterations: int = DEFAULT_OPTSPACE_ITERATIONS) -> (
        OrdinationResults,
        DistanceMatrix,
        pd.DataFrame):
    """
    Performs joint-RPCA across data tables
    with shared samples.

    Parameters
    ----------
    tables: list of biom.Table, required
    A list of feature table in biom format containing shared
    samples over which metric should be computed.

    n_test_samples: int, optional : Default is 10
    Number of random samples to choose for test split samples.

    metadata: DataFrame, optional : Default is None
    Sample metadata file in QIIME2 formatting. The file must
    contain a train-test column with labels `train` and `test`
    and the row ids matched to the table(s).

    train_test_column: str, optional : Default is None
    Sample metadata column containing `train` and `test`
    labels to use for the cross-validation evaluation.

    n_components: int, optional : Default is 3
    The underlying rank of the data and number of
    output dimensions.

    rclr_transform_tables: bool, optional: If False Joint-RPCA
    will not use the RCLR transformation and will instead
    assume that the data has already been transformed
    or normalized. Default is True.

    max_iterations: int, optional : Default is 5
    The number of convex iterations to optimize the solution
    If iteration is not specified, then the default iteration is 5.
    Which reduces to a satisfactory error threshold.

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

    Returns
    -------
    OrdinationResults
        A joint-biplot of the (Robust Aitchison) RPCA feature loadings

    DistanceMatrix
        The Aitchison distance of the sample loadings from RPCA.

    DataFrame
        The cross-validation reconstruction error.

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

    """

    # filter each table
    for n, table_n in enumerate(tables):
        if rclr_transform_tables:
            tables[n] = rpca_table_processing(table_n,
                                              min_sample_count,
                                              min_feature_count,
                                              min_feature_frequency)
    # get set of shared samples
    shared_all_samples = set.intersection(*[set(table_n.ids())
                                            for table_n in tables])
    # check sample overlaps
    if len(shared_all_samples) == 0:
        raise ValueError('No samples overlap between all tables. '
                         'If using pre-transformed or normalized '
                         'tables, make sure the rclr_transform_tables '
                         'is set to False or the flag is enabled.')
    unshared_samples = set([s_n
                            for table_n in tables
                            for s_n in table_n.ids()]) - shared_all_samples
    if len(unshared_samples) != 0:
        warnings.warn('Removing %i sample(s) that do not overlap in tables.'
                      % (len(unshared_samples)), RuntimeWarning)
    # filter each table again to subset samples.
    for n, table_n in enumerate(tables):
        if rclr_transform_tables:
            table_n = table_n.filter(shared_all_samples)
            tables[n] = rpca_table_processing(table_n,
                                              min_sample_count,
                                              min_feature_count,
                                              min_feature_frequency)
        else:
            tables[n] = table_n.filter(shared_all_samples)
    shared_all_samples = set.intersection(*[set(table_n.ids())
                                            for table_n in tables])
    # rclr each table
    rclr_tables = []
    for table_n in tables:
        # perform RCLR
        if rclr_transform_tables:
            rclr_tmp = matrix_rclr(table_n.matrix_data.toarray().T).T
        # otherwise just mask zeros
        else:
            rclr_tmp = mask_value_only(table_n.matrix_data.toarray().T).T
        rclr_tables.append(pd.DataFrame(rclr_tmp,
                                        table_n.ids('observation'),
                                        table_n.ids()))
    # get training and test sample IDs
    if sample_metadata is not None and not isinstance(sample_metadata,
                                                      pd.DataFrame):
        sample_metadata = sample_metadata.to_dataframe()
    if sample_metadata is None or train_test_column is None:
        test_samples = sorted(list(shared_all_samples))[:n_test_samples]
        train_samples = list(set(shared_all_samples) - set(test_samples))
    else:
        sample_metadata = sample_metadata.loc[shared_all_samples, :]
        train_samples = sample_metadata[train_test_column] == 'train'
        test_samples = sample_metadata[train_test_column] == 'test'
        train_samples = sample_metadata[train_samples].index
        test_samples = sample_metadata[test_samples].index
    ord_res, U_dist_res, cv_dist = joint_optspace_helper(rclr_tables,
                                                         n_components,
                                                         max_iterations,
                                                         test_samples,
                                                         train_samples)
    return ord_res, U_dist_res, cv_dist


def joint_optspace_helper(tables,
                          n_components,
                          max_iterations,
                          test_samples,
                          train_samples):
    """
    Helper function for joint-RPCA
    """

    # split the tables by training and test samples
    tables_split = [[table_i.loc[:, test_samples].T,
                     table_i.loc[:, train_samples].T]
                    for table_i in tables]
    # run OptSpace
    opt_model = OptSpace(n_components=n_components,
                         max_iterations=max_iterations,
                         tol=None)
    U, s, Vs, dists = opt_model.joint_solve([[t_s.values for t_s in t]
                                             for t in tables_split])
    rename_cols = ['PC' + str(i + 1) for i in range(n_components)]
    vjoint = pd.concat([pd.DataFrame(Vs_n,
                                     index=t_n.index,
                                     columns=rename_cols)
                        for t_n, Vs_n in zip(tables, Vs)])
    ujoint = pd.DataFrame(U,
                          index=list(train_samples),
                          columns=rename_cols)
    # center again around zero after completion
    X = ujoint.values @ s @ vjoint.values.T
    X = X - X.mean(axis=0)
    X = X - X.mean(axis=1).reshape(-1, 1)
    u, s_new, v = svd(X, full_matrices=False)
    s_eig = s_new[:n_components]
    rename_cols = ['PC' + str(i + 1) for i in range(n_components)]
    v = v.T[:, :n_components]
    u = u[:, :n_components]
    # create ordination
    vjoint = pd.DataFrame(v,
                          index=vjoint.index,
                          columns=vjoint.columns)
    ujoint = pd.DataFrame(u,
                          index=list(train_samples),
                          columns=ujoint.columns)
    p = s_eig**2 / np.sum(s_eig**2)
    eigvals = pd.Series(s_eig, index=rename_cols)
    proportion_explained = pd.Series(p, index=rename_cols)
    ord_res = OrdinationResults(
            'rpca',
            'rpca',
            eigvals.copy(),
            samples=ujoint.copy(),
            features=vjoint.copy(),
            proportion_explained=proportion_explained.copy())
    # project test data into training data
    if len(test_samples) > 0:
        ord_res = transform(ord_res,
                            [t[0] for t in tables_split],
                            rclr_transform=False)
    # save results
    Udist = distance.cdist(ord_res.samples.copy(),
                           ord_res.samples.copy())
    U_dist_res = DistanceMatrix(Udist, ids=ord_res.samples.index)
    cv_dist = pd.DataFrame(dists, ['mean_CV', 'std_CV']).T
    cv_dist['run'] = 'tables_%i.n_components_%i.max_iterations_%i.n_test_%i' \
                     % (len(tables), n_components,
                        max_iterations, len(test_samples))
    cv_dist['iteration'] = list(cv_dist.index.astype(int))
    cv_dist.index.name = 'sampleid'

    return ord_res, U_dist_res, cv_dist


def transform(ordination: OrdinationResults,
              tables: biom.Table,
              subset_tables: bool = DEFAULT_MATCH,
              rclr_transform: bool = DEFAULT_TRNSFRM) -> (
        OrdinationResults):
    """
    Function to apply dimensionality reduction to table(s).
    The table(s) is projected on the first principal components
    previously extracted from a training set.

    Parameters
    ----------
    ordination: OrdinationResults
        A joint-biplot of the (Robust Aitchison) RPCA feature loadings
        produced from the training data.

    tables: list of biom.Table, required
        A list of at least one feature table in biom format containing
        shared samples over which metric should be computed.

    subset_tables: bool, optional : default is True
        Subsets the input tables to contain only features used in the
        training data. If set to False and the tables are not perfectly
        matched a ValueError will be produced.

    rclr_transform: bool, optional : default is True
        If set to false the function will expect `tables` to be dataframes
        already rclr transformed. This is used for internal functionality
        in the joint-rpca function.

    Returns
    -------
    OrdinationResults
        A joint-biplot of the (Robust Aitchison) RPCA feature loadings
        with both the input training data and new test data.

    Raises
    ------
    ValueError
        `ValueError: The input tables do not contain all
        the features in the ordination.`.

    ValueError
        `ValueError: Removing # features(s) in table(s)
        but not the ordination.`.

    ValueError
        `ValueError: Features in the input table(s) not in
        the features in the ordination.  Either set subset_tables to
        True or match the tables to the ordination.`.
    """

    # extract current U & V matrix
    Udf = ordination.samples.copy()
    Vdf = ordination.features.copy()
    s_eig = ordination.eigvals.copy().values
    # rclr each table [if needed]
    rclr_table_df = []
    if rclr_transform:
        for table_n in tables:
            rclr_tmp = matrix_rclr(table_n.matrix_data.toarray().T)
            rclr_table_df.append(pd.DataFrame(rclr_tmp,
                                              table_n.ids(),
                                              table_n.ids('observation')))
    else:
        for table_n in tables:
            rclr_table_df.append(table_n)
    rclr_table_df = pd.concat(rclr_table_df, axis=1).T
    # ensure feature IDs match
    shared_features = set(rclr_table_df.index) & set(Vdf.index)
    if len(shared_features) < len(set(Vdf.index)):
        raise ValueError('The input tables do not contain all'
                         ' the features in the ordination.')
    elif subset_tables:
        unshared_N = len(set(rclr_table_df.index)) - len(shared_features)
        warnings.warn('Removing %i features(s) in table(s)'
                      ' but not the ordination.'
                      % (unshared_N), RuntimeWarning)
    else:
        raise ValueError('Features in the input table(s) not in'
                         ' the features in the ordination.'
                         ' Either set subset_tables to True or'
                         ' match the tables to the ordination.')
    ordination.samples = transform_helper(Udf,
                                          Vdf,
                                          s_eig,
                                          rclr_table_df)
    return ordination


def transform_helper(Udf, Vdf, s_eig, table_rclr_project):
    # project new data into ordination
    table_rclr_project = table_rclr_project.reindex(Vdf.index)
    M_project = np.ma.array(table_rclr_project,
                            mask=np.isnan(table_rclr_project)).T
    M_project = M_project - M_project.mean(axis=1).reshape(-1, 1)
    M_project = M_project - M_project.mean(axis=0)
    U_projected = np.ma.dot(M_project, Vdf.values).data
    U_projected /= np.linalg.norm(s_eig)
    U_projected = pd.DataFrame(U_projected,
                               table_rclr_project.columns,
                               Udf.columns)
    return pd.concat([Udf, U_projected])


def rpca_transform(ordination: OrdinationResults,
                   table: biom.Table,
                   subset_tables: bool = DEFAULT_MATCH,
                   rclr_transform: bool = DEFAULT_TRNSFRM) -> (
        OrdinationResults):
    """
    To avoid confusion this helper function takes one input
    to use in QIIME2.
    """
    ordination = transform(ordination, [table],
                           subset_tables=subset_tables,
                           rclr_transform=rclr_transform)
    return ordination


def feature_covariance_table(ordination, features_use=None):
    """
    Function to produce a feature by feature
    covariance table from RPCA ordination
    results.

    Parameters
    ----------
    ordination: OrdinationResults
        A joint-biplot of the (Robust Aitchison) RPCA feature loadings
    features_use: list, optional : default is None
        A subset of features to use in the covariance generation.

    Returns
    -------
    DataFrame
        A feature by feature covariance table.

    """
    if features_use is not None:
        vjoint = ordination.features.copy()
        if len(set(features_use) - set(vjoint.index)) != 0:
            raise ValueError('Feature subset given contains labels'
                             ' not in the loadings.')
        vjoint = vjoint.loc[features_use, :]
    else:
        vjoint = ordination.features
    s = ordination.eigvals.values
    Vs_joint = vjoint.values @ np.diag(s)**2 @ vjoint.values.T
    joint_features = pd.DataFrame(Vs_joint,
                                  vjoint.index,
                                  vjoint.index)

    return joint_features


def feature_correlation_table(ordination: OrdinationResults) -> (pd.DataFrame):
    """
    Function to produce a feature by feature
    correlation table from RPCA ordination
    results. Note that the output can be very large in
    file size because it is all omics features by all
    omics features and is fully dense. If you would like to
    get a subset, just subset the ordination with the function
    `filter_ordination` in utils first.

    Parameters
    ----------
    ordination: OrdinationResults
        A joint-biplot of the (Robust Aitchison) RPCA feature loadings.

    Returns
    -------
    DataFrame
        A feature by feature correlation table.

    """
    joint_features = feature_covariance_table(ordination)
    # this part of the function is taken from:
    # https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
    covariance = joint_features.values
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    # convert back to dataframe
    correlation = pd.DataFrame(correlation,
                               joint_features.index,
                               joint_features.columns)
    correlation.index.name = 'featureid'
    return correlation
