import biom
import skbio
import numpy as np
import pandas as pd
from pandas import concat
from pandas import DataFrame
from typing import Optional
from skbio import OrdinationResults, DistanceMatrix, TreeNode
from gemelli.factorization import TensorFactorization
from gemelli.rpca import rpca_table_processing
from gemelli.preprocessing import (build,
                                   fast_unifrac,
                                   bp_read_phylogeny,
                                   retrieve_t2t_taxonomy,
                                   create_taxonomy_metadata)
from gemelli._defaults import (DEFAULT_COMP, DEFAULT_MSC,
                               DEFAULT_MFC, DEFAULT_BL,
                               DEFAULT_MTD, DEFAULT_MFF,
                               DEFAULT_TENSALS_MAXITER,
                               DEFAULT_FMETA as DEFFM)
# import QIIME2 if in a Q2env otherwise set type to str
try:
    from q2_types.tree import NewickFormat
except ImportError:
    # python does not check but technically this is the type
    NewickFormat = str


def phylogenetic_ctf_without_taxonomy(
                     table: biom.Table,
                     phylogeny: NewickFormat,
                     sample_metadata: DataFrame,
                     individual_id_column: str,
                     state_column: str,
                     n_components: int = DEFAULT_COMP,
                     min_sample_count: int = DEFAULT_MSC,
                     min_feature_count: int = DEFAULT_MFC,
                     min_feature_frequency: float = DEFAULT_MFF,
                     min_depth: int = DEFAULT_MTD,
                     max_iterations_als: int = DEFAULT_TENSALS_MAXITER,
                     max_iterations_rptm: int = DEFAULT_TENSALS_MAXITER,
                     n_initializations: int = DEFAULT_TENSALS_MAXITER) -> (
                         OrdinationResults, OrdinationResults,
                         DistanceMatrix, DataFrame, DataFrame,
                         TreeNode, biom.Table, biom.Table):
    """
    Runs phylogenetic CTF without taxonomy. This code will
    be run QIIME2 versions of gemelli. Outside of QIIME2
    please use phylogenetic_ctf.
    """
    # run CTF helper and parse output for QIIME
    output = phylogenetic_ctf(table=table,
                              phylogeny=phylogeny,
                              sample_metadata=sample_metadata,
                              individual_id_column=individual_id_column,
                              state_column=state_column,
                              n_components=n_components,
                              min_sample_count=min_sample_count,
                              min_feature_count=min_feature_count,
                              min_feature_frequency=min_feature_frequency,
                              min_depth=min_depth,
                              max_iterations_als=max_iterations_als,
                              max_iterations_rptm=max_iterations_rptm,
                              n_initializations=n_initializations)
    (ord_res, state_ordn, dists, straj, ftraj,
     phylogeny, counts_by_node,
     _, subject_table) = output

    return (ord_res, state_ordn, dists, straj, ftraj,
            phylogeny, counts_by_node, subject_table)


def phylogenetic_ctf_with_taxonomy(
                     table: biom.Table,
                     phylogeny: NewickFormat,
                     taxonomy: pd.DataFrame,
                     sample_metadata: DataFrame,
                     individual_id_column: str,
                     state_column: str,
                     n_components: int = DEFAULT_COMP,
                     min_sample_count: int = DEFAULT_MSC,
                     min_feature_count: int = DEFAULT_MFC,
                     min_feature_frequency: float = DEFAULT_MFF,
                     min_depth: int = DEFAULT_MTD,
                     max_iterations_als: int = DEFAULT_TENSALS_MAXITER,
                     max_iterations_rptm: int = DEFAULT_TENSALS_MAXITER,
                     n_initializations: int = DEFAULT_TENSALS_MAXITER) -> (
                         OrdinationResults, OrdinationResults,
                         DistanceMatrix, DataFrame, DataFrame,
                         TreeNode, biom.Table,
                         pd.DataFrame, biom.Table):
    """
    Runs phylogenetic CTF with taxonomy. This code will
    be run QIIME2 versions of gemelli. Outside of QIIME2
    please use phylogenetic_ctf.
    """
    # feature_metadata
    taxonomy = taxonomy.to_dataframe()
    # run CTF helper and parse output for QIIME
    output = phylogenetic_ctf(table=table,
                              phylogeny=phylogeny,
                              taxonomy=taxonomy,
                              sample_metadata=sample_metadata,
                              individual_id_column=individual_id_column,
                              state_column=state_column,
                              n_components=n_components,
                              min_sample_count=min_sample_count,
                              min_feature_count=min_feature_count,
                              min_feature_frequency=min_feature_frequency,
                              min_depth=min_depth,
                              max_iterations_als=max_iterations_als,
                              max_iterations_rptm=max_iterations_rptm,
                              n_initializations=n_initializations)
    (ord_res, state_ordn, dists, straj, ftraj,
     phylogeny, counts_by_node,
     result_taxonomy, subject_table) = output

    return (ord_res, state_ordn,
            dists, straj, ftraj,
            phylogeny, counts_by_node,
            result_taxonomy, subject_table)


def phylogenetic_ctf(table: biom.Table,
                     phylogeny: NewickFormat,
                     sample_metadata: DataFrame,
                     individual_id_column: str,
                     state_column: str,
                     taxonomy: Optional[pd.DataFrame] = None,
                     n_components: int = DEFAULT_COMP,
                     min_sample_count: int = DEFAULT_MSC,
                     min_feature_count: int = DEFAULT_MFC,
                     min_feature_frequency: float = DEFAULT_MFF,
                     min_depth: int = DEFAULT_MTD,
                     max_iterations_als: int = DEFAULT_TENSALS_MAXITER,
                     max_iterations_rptm: int = DEFAULT_TENSALS_MAXITER,
                     n_initializations: int = DEFAULT_TENSALS_MAXITER) -> (
                         OrdinationResults, OrdinationResults,
                         DistanceMatrix, DataFrame, DataFrame,
                         TreeNode, biom.Table,
                         Optional[pd.DataFrame], biom.Table):
    """
    Phylogenetic Compositional Tensor Factorization (CTF) with
    mode 3 tensor. This means subjects have repeated measures
    across only one axis (e.g. time or space).
    For more information see (1-4).

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

    sample_metadata: DataFrame, required
    Sample metadata file in QIIME2 formatting. The file must
    contain the columns for individual_id_column and
    state_column and the rows matched to the table.

    individual_id_column: str, required
    Metadata column containing subject IDs to use for
    pairing samples. WARNING: if replicates exist for an
    individual ID at either state_1 to state_N, that
    subject will be mean grouped by default.

    state_column: str, required
    Metadata column containing state (e.g.,Time,
    BodySite) across which samples are paired. At least
    one is required but up to four are allowed by other
    state inputs.

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

    max_als_iterations: int, optional
    Max number of Alternating Least Square (ALS).

    tol_als: float, optional
    The minimization -- convergence break point for ALS.

    max_rtpm_iterations: int, optional
    Max number of Robust Tensor Power Method (RTPM) iterations.

    n_initializations: int, optional
    The number of initial vectors. Larger values will
    give more accurate factorization but will be more
    computationally expensive.

    Returns
    -------
    OrdinationResults
        Compositional biplot of subjects as points and
        features as arrows. Where the variation between
        subject groupings is explained by the log-ratio
        between opposing arrows. WARNING: The % variance
        explained is only spread over n-components and
        can be inflated.

    OrdinationResults
        Compositional biplot of states as points and
        features as arrows. Where the variation between
        subject groupings is explained by the log-ratio
        between opposing arrows. WARNING: The % variance
        explained is only spread over n-components and can be
        inflated.

    DistanceMatrix
        A sample-sample distance matrix generated from the
        euclidean distance of the subject-state ordinations
        and itself.

    DataFrame
        A trajectory is an ordination that can be
        visualizedover time or another context.

    DataFrame
        A trajectory is an ordination that can be
        visualizedover time or another context.

    Raises
    ------
    ValueError
        `ValueError: n_components must be at least 2`.

    ValueError
        `ValueError: Data-table contains either np.inf or -np.inf`.

    ValueError
        `ValueError: The n_components must be less
            than the minimum shape of the input tensor`.

    References
    ----------
    .. [1] Martino C, Shenhav L, Marotz CA, Armstrong G, McDonald D,
           Vázquez-Baeza Y, Morton JT, Jiang L, Dominguez-Bello MG,
           Swafford AD, Halperin E, Knight R. 2020.
           Context-aware dimensionality reduction deconvolutes
           gut microbial community dynamics.
           Nat Biotechnol https://doi.org/10.1038/s41587-020-0660-7.
    .. [2] Jain, Prateek, and Sewoong Oh. 2014.
            “Provable Tensor Factorization with Missing Data.”
            In Advances in Neural Information Processing Systems
            27, edited by Z. Ghahramani, M. Welling, C. Cortes,
            N. D. Lawrence, and K. Q. Weinberger, 1431–39.
            Curran Associates, Inc.
    .. [3] A. Anandkumar, R. Ge, M., Janzamin,
            Guaranteed Non-Orthogonal Tensor
            Decomposition via Alternating Rank-1
            Updates. CoRR (2014),
            pp. 1-36.
    .. [4] A. Anandkumar, R. Ge, D. Hsu,
            S. M. Kakade, M. Telgarsky,
            Tensor Decompositions for Learning Latent Variable Models
            (A Survey for ALT). Lecture Notes in Computer Science
            (2015), pp. 19–38.

    Examples
    --------
    import numpy as np
    import pandas as pd
    from biom import Table
    from gemelli.ctf import phylogenetic_ctf

    # make a table
    X = np.array([[9, 3, 0, 0],
                [9, 9, 0, 1],
                [0, 1, 4, 5],
                [0, 0, 3, 4],
                [1, 0, 8, 9]])
    sample_ids = ['s1','s2','s3','s4']
    feature_ids = ['f1','f2','f3','f4','f5']
    bt = Table(X, feature_ids, sample_ids)
    # make mock metadata
    mf = pd.DataFrame([[i//2, i%2] for i, s in enumerate(sample_ids)],
                    sample_ids, ['subject_id', 'context'])
    # write an example tree to read
    f = open("demo-tree.nwk", "w")
    newick = '(((f1:1,f2:1)n9:1,f3:1)n8:1,(f4:1,f5:1)n2:1)n1:1;'
    f.write(newick)
    f.close()
    # make mock taxonomy
    taxonomy = pd.DataFrame({fid:['k__kingdom; p__phylum;'
                                'c__class; o__order; '
                                'f__family; g__genus;'
                                's__',
                                0.99]
                            for fid in feature_ids},
                            ['Taxon', 'Confidence']).T

    # run phylo-CTF with taxonomy
    # subject 1 will seperate from subject 2
    (subject_biplot, state_biplot,
    distance_matrix,
    state_subject_ordination,
    state_feature_ordination,
    counts_by_node_tree,
    counts_by_node,
    _,
    subject_table) = phylogenetic_ctf(bt, 'demo-tree.nwk', mf,
                                    'subject_id', 'context')


    # run phylo-CTF without taxonomy
    # subject 1 will seperate from subject 2
    (subject_biplot, state_biplot,
    distance_matrix,
    state_subject_ordination,
    state_feature_ordination,
    counts_by_node_tree,
    counts_by_node,
    t2t_taxonomy,
    subject_table) = phylogenetic_ctf(bt, 'demo-tree.nwk', mf,
                                    'subject_id', 'context', taxonomy)

    """

    # run CTF helper and parse output for QIIME
    helper_results = phylogenetic_ctf_helper(table,
                                             phylogeny,
                                             sample_metadata,
                                             individual_id_column,
                                             [state_column],
                                             n_components,
                                             min_sample_count,
                                             min_feature_count,
                                             min_feature_frequency,
                                             min_depth,
                                             max_iterations_als,
                                             max_iterations_rptm,
                                             n_initializations,
                                             taxonomy)
    (state_ordn, ord_res, dists, straj,
     ftraj, phylogeny, counts_by_node,
     result_taxonomy, subject_table) = helper_results

    # save only first state (QIIME can't handle a list yet)
    dists = list(dists.values())[0]
    straj = list(straj.values())[0]
    ftraj = list(ftraj.values())[0]
    state_ordn = list(state_ordn.values())[0]

    return (ord_res, state_ordn,
            dists, straj, ftraj,
            phylogeny, counts_by_node,
            result_taxonomy, subject_table)


def phylogenetic_ctf_helper(table: biom.Table,
                            phylogeny: NewickFormat,
                            sample_metadata: DataFrame,
                            individual_id_column: str,
                            state_column: list,
                            n_components: int = DEFAULT_COMP,
                            min_sample_count: int = DEFAULT_MSC,
                            min_feature_count: int = DEFAULT_MFC,
                            min_feature_frequency: float = DEFAULT_MFF,
                            min_depth: int = DEFAULT_MTD,
                            max_iterations_als: int = DEFAULT_TENSALS_MAXITER,
                            max_iterations_rptm: int = DEFAULT_TENSALS_MAXITER,
                            n_initializations: int = DEFAULT_TENSALS_MAXITER,
                            taxonomy: Optional[pd.DataFrame] = None) -> (
                                OrdinationResults, OrdinationResults,
                                DistanceMatrix, DataFrame, DataFrame,
                                TreeNode, biom.Table, biom.Table):
    """Helper function. Please use phylogenetic_ctf directly."""
    # check the table for validity and then filter
    process_results = ctf_table_processing(table,
                                           sample_metadata,
                                           individual_id_column,
                                           state_column,
                                           min_sample_count,
                                           min_feature_count,
                                           min_feature_frequency,
                                           taxonomy)
    (table, sample_metadata,
     all_sample_metadata, taxonomy) = process_results
    # import the tree
    phylogeny = bp_read_phylogeny(table, phylogeny, min_depth)
    # build the vectorized table
    counts_by_node, tree_index, branch_lengths, fids, otu_ids\
        = fast_unifrac(table, phylogeny)
    # import expanded table
    counts_by_node = biom.Table(counts_by_node.T,
                                fids, table.ids())
    # make a table for Empress
    subject_table = per_subject_table(table,
                                      sample_metadata.copy(),
                                      individual_id_column)
    # use taxonomy if provided
    result_taxonomy = None
    if taxonomy is not None:
        # collect taxonomic information for all tree nodes.
        traversed_taxonomy = retrieve_t2t_taxonomy(phylogeny, taxonomy)
        result_taxonomy = create_taxonomy_metadata(phylogeny,
                                                   traversed_taxonomy)
    # build the tensor object and factor - return results
    tensal_results = tensals_helper(counts_by_node,
                                    sample_metadata,
                                    all_sample_metadata,
                                    individual_id_column,
                                    state_column,
                                    branch_lengths,
                                    n_components,
                                    max_iterations_als,
                                    max_iterations_rptm,
                                    n_initializations,
                                    result_taxonomy)
    state_ordn, ord_res, dists, straj, ftraj = tensal_results

    return (state_ordn, ord_res,
            dists, straj, ftraj,
            phylogeny, counts_by_node,
            result_taxonomy, subject_table)


def ctf(table: biom.Table,
        sample_metadata: DataFrame,
        individual_id_column: str,
        state_column: str,
        n_components: int = DEFAULT_COMP,
        min_sample_count: int = DEFAULT_MSC,
        min_feature_count: int = DEFAULT_MFC,
        min_feature_frequency: float = DEFAULT_MFF,
        max_iterations_als: int = DEFAULT_TENSALS_MAXITER,
        max_iterations_rptm: int = DEFAULT_TENSALS_MAXITER,
        n_initializations: int = DEFAULT_TENSALS_MAXITER,
        feature_metadata: DataFrame = DEFFM) -> (OrdinationResults,
                                                 OrdinationResults,
                                                 DistanceMatrix,
                                                 DataFrame,
                                                 DataFrame):
    """
    Compositional Tensor Factorization (CTF) with mode 3 tensor. This
    means subjects have repeated measures across only one axis
    (e.g. time or space). For more information see (1-4).

    Parameters
    ----------
    table: numpy.ndarray, required
    The feature table in biom format containing the
    samples over which metric should be computed.

    sample_metadata: DataFrame, required
    Sample metadata file in QIIME2 formatting. The file must
    contain the columns for individual_id_column and
    state_column and the rows matched to the table.

    individual_id_column: str, required
    Metadata column containing subject IDs to use for
    pairing samples. WARNING: if replicates exist for an
    individual ID at either state_1 to state_N, that
    subject will be mean grouped by default.

    state_column: str, required
    Metadata column containing state (e.g.,Time,
    BodySite) across which samples are paired. At least
    one is required but up to four are allowed by other
    state inputs.

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

    max_als_iterations: int, optional
    Max number of Alternating Least Square (ALS).

    tol_als: float, optional
    The minimization -- convergence break point for ALS.

    max_rtpm_iterations: int, optional
    Max number of Robust Tensor Power Method (RTPM) iterations.

    n_initializations: int, optional
    The number of initial vectors. Larger values will
    give more accurate factorization but will be more
    computationally expensive.

    feature_metadata: pd.DataFrame, optional
    Taxonomy file in QIIME2 formatting. See the feature metdata
    section of https://docs.qiime2.org/2021.11/tutorials/metadata

    Returns
    -------
    OrdinationResults
        Compositional biplot of subjects as points and
        features as arrows. Where the variation between
        subject groupings is explained by the log-ratio
        between opposing arrows. WARNING: The % variance
        explained is only spread over n-components and
        can be inflated.

    OrdinationResults
        Compositional biplot of states as points and
        features as arrows. Where the variation between
        subject groupings is explained by the log-ratio
        between opposing arrows. WARNING: The % variance
        explained is only spread over n-components and can be
        inflated.

    DistanceMatrix
        A sample-sample distance matrix generated from the
        euclidean distance of the subject-state ordinations
        and itself.

    DataFrame
        A trajectory is an ordination that can be
        visualizedover time or another context.

    DataFrame
        A trajectory is an ordination that can be
        visualizedover time or another context.

    Raises
    ------
    ValueError
        `ValueError: n_components must be at least 2`.

    ValueError
        `ValueError: Data-table contains either np.inf or -np.inf`.

    ValueError
        `ValueError: The n_components must be less
            than the minimum shape of the input tensor`.

    References
    ----------
    .. [1] Martino C, Shenhav L, Marotz CA, Armstrong G, McDonald D,
           Vázquez-Baeza Y, Morton JT, Jiang L, Dominguez-Bello MG,
           Swafford AD, Halperin E, Knight R. 2020.
           Context-aware dimensionality reduction deconvolutes
           gut microbial community dynamics.
           Nat Biotechnol https://doi.org/10.1038/s41587-020-0660-7.
    .. [2] Jain, Prateek, and Sewoong Oh. 2014.
            “Provable Tensor Factorization with Missing Data.”
            In Advances in Neural Information Processing Systems
            27, edited by Z. Ghahramani, M. Welling, C. Cortes,
            N. D. Lawrence, and K. Q. Weinberger, 1431–39.
            Curran Associates, Inc.
    .. [3] A. Anandkumar, R. Ge, M., Janzamin,
            Guaranteed Non-Orthogonal Tensor
            Decomposition via Alternating Rank-1
            Updates. CoRR (2014),
            pp. 1-36.
    .. [4] A. Anandkumar, R. Ge, D. Hsu,
            S. M. Kakade, M. Telgarsky,
            Tensor Decompositions for Learning Latent Variable Models
            (A Survey for ALT). Lecture Notes in Computer Science
            (2015), pp. 19–38.

    Examples
    --------
    import numpy as np
    import pandas as pd
    from biom import Table
    from gemelli.ctf import ctf

    # make a table
    X = np.array([[9, 3, 0, 0],
                [9, 9, 0, 1],
                [0, 1, 4, 5],
                [0, 0, 3, 4],
                [1, 0, 8, 9]])
    sample_ids = ['s1','s2','s3','s4']
    feature_ids = ['f1','f2','f3','f4','f5']
    bt = Table(X, feature_ids, sample_ids)
    # make mock metadata
    mf = pd.DataFrame([[i//2, i%2] for i, s in enumerate(sample_ids)],
                    sample_ids, ['subject_id', 'context'])
    # run CTF
    # subject 1 will seperate from subject 2
    (subject_biplot, state_biplot,
    distance_matrix,
    state_subject_ordination,
    state_feature_ordination) = ctf(bt, mf, 'subject_id', 'context')

    """
    # run CTF helper and parse output for QIIME
    helper_results = ctf_helper(table,
                                sample_metadata,
                                individual_id_column,
                                [state_column],
                                n_components,
                                min_sample_count,
                                min_feature_count,
                                min_feature_frequency,
                                max_iterations_als,
                                max_iterations_rptm,
                                n_initializations,
                                feature_metadata)
    state_ordn, ord_res, dists, straj, ftraj = helper_results
    # save only first state (QIIME can't handle a list yet)
    dists = list(dists.values())[0]
    straj = list(straj.values())[0]
    ftraj = list(ftraj.values())[0]
    state_ordn = list(state_ordn.values())[0]

    return ord_res, state_ordn, dists, straj, ftraj


def ctf_helper(table: biom.Table,
               sample_metadata: DataFrame,
               individual_id_column: str,
               state_column: list,
               n_components: int = DEFAULT_COMP,
               min_sample_count: int = DEFAULT_MSC,
               min_feature_count: int = DEFAULT_MFC,
               min_feature_frequency: float = DEFAULT_MFF,
               max_iterations_als: int = DEFAULT_TENSALS_MAXITER,
               max_iterations_rptm: int = DEFAULT_TENSALS_MAXITER,
               n_initializations: int = DEFAULT_TENSALS_MAXITER,
               feature_metadata: DataFrame = DEFFM) -> (
                   OrdinationResults, OrdinationResults,
                   DistanceMatrix, DataFrame, DataFrame):
    """Helper function. Please use ctf directly."""
    # check the table for validity and then filter
    process_results = ctf_table_processing(table,
                                           sample_metadata,
                                           individual_id_column,
                                           state_column,
                                           min_sample_count,
                                           min_feature_count,
                                           min_feature_frequency,
                                           feature_metadata)
    (table, sample_metadata,
     all_sample_metadata, feature_metadata) = process_results
    # build the tensor object and factor - return results
    tensal_results = tensals_helper(table,
                                    sample_metadata,
                                    all_sample_metadata,
                                    individual_id_column,
                                    state_column,
                                    None,
                                    n_components,
                                    max_iterations_als,
                                    max_iterations_rptm,
                                    n_initializations,
                                    feature_metadata)
    state_ordn, ord_res, dists, straj, ftraj = tensal_results

    return state_ordn, ord_res, dists, straj, ftraj


def per_subject_table(table: biom.Table,
                      sample_metadata: DataFrame,
                      individual_id_column: str):
    """ builds a per-subject summed table for Empress
    """

    # convert biom.Table into dataframe
    subject_table = DataFrame(table.matrix_data.toarray(),
                              table.ids('observation'),
                              table.ids('sample')).T
    # get subject ID information
    subject_sample_mf = sample_metadata.copy()[individual_id_column]
    subject_sample_mf = subject_sample_mf.reindex(subject_table.index)
    subject_table[individual_id_column] = subject_sample_mf
    # sum by subject across all samples
    subject_table = subject_table.groupby(individual_id_column).sum().T
    # back to biom.Table
    subject_table = biom.Table(subject_table.values,
                               subject_table.index.astype(str),
                               subject_table.columns.astype(str))

    return subject_table


def ctf_table_processing(table: biom.Table,
                         sample_metadata: DataFrame,
                         individual_id_column: str,
                         state_columns: list,
                         min_sample_count: int = DEFAULT_MSC,
                         min_feature_count: int = DEFAULT_MFC,
                         min_feature_frequency: float = DEFAULT_MFF,
                         feature_metadata: DataFrame = DEFFM) -> (
                             dict, OrdinationResults, dict, tuple):
    """ Runs  Compositional Tensor Factorization CTF.
    """

    # validate the metadata using q2 as a wrapper
    if sample_metadata is not None and not isinstance(sample_metadata,
                                                      DataFrame):
        sample_metadata = sample_metadata.to_dataframe()
    keep_cols = state_columns + [individual_id_column]
    all_sample_metadata = sample_metadata.drop(keep_cols, axis=1)
    sample_metadata = sample_metadata[keep_cols]
    # validate the metadata using q2 as a wrapper
    if feature_metadata is not None and not isinstance(feature_metadata,
                                                       DataFrame):
        feature_metadata = feature_metadata.to_dataframe()
    # match the data (borrowed in part from gneiss.util.match)
    subtablefids = table.ids('observation')
    subtablesids = table.ids('sample')
    if len(subtablesids) != len(set(subtablesids)):
        raise ValueError('Data-table contains duplicate sample IDs')
    if len(subtablefids) != len(set(subtablefids)):
        raise ValueError('Data-table contains duplicate feature IDs')
    submetadataids = set(sample_metadata.index)
    subtablesids = set(subtablesids)
    subtablefids = set(subtablefids)
    if feature_metadata is not None:
        submetadatafeat = set(feature_metadata.index)
        fidx = subtablefids & submetadatafeat
        if len(fidx) == 0:
            raise ValueError(("No more features left.  Check to make "
                              "sure that the sample names between "
                              "`feature-metadata` and `table` are "
                              "consistent"))
        feature_metadata = feature_metadata.reindex(fidx)
    sidx = subtablesids & submetadataids
    if len(sidx) == 0:
        raise ValueError(("No more features left.  Check to make sure that "
                          "the sample names between `sample-metadata` and"
                          " `table` are consistent"))
    if feature_metadata is not None:
        table.filter(list(fidx), axis='observation', inplace=True)
    table.filter(list(sidx), axis='sample', inplace=True)
    sample_metadata = sample_metadata.reindex(sidx)

    # filter the table (same as RPCA)
    table = rpca_table_processing(table,
                                  min_sample_count,
                                  min_feature_count,
                                  min_feature_frequency)

    # return data based on input
    if feature_metadata is not None:
        return (table, sample_metadata,
                all_sample_metadata,
                feature_metadata)
    else:
        return (table, sample_metadata,
                all_sample_metadata, None)


def tensals_helper(table: biom.Table,
                   sample_metadata: DataFrame,
                   all_sample_metadata: DataFrame,
                   individual_id_column: str,
                   state_columns: list,
                   branch_lengths: np.array = DEFAULT_BL,
                   n_components: int = DEFAULT_COMP,
                   max_iterations_als: int = DEFAULT_TENSALS_MAXITER,
                   max_iterations_rptm: int = DEFAULT_TENSALS_MAXITER,
                   n_initializations: int = DEFAULT_TENSALS_MAXITER,
                   feature_metadata: DataFrame = DEFFM) -> (
                       dict, OrdinationResults, dict, tuple):
    """ Runs  Compositional Tensor Factorization CTF.
    """

    # table to dataframe
    table = DataFrame(table.matrix_data.toarray(),
                      table.ids('observation'),
                      table.ids('sample'))

    # tensor building
    tensor = build()
    tensor.construct(table, sample_metadata,
                     individual_id_column,
                     state_columns,
                     branch_lengths=branch_lengths)

    # factorize
    TF = TensorFactorization(
        n_components=n_components,
        max_als_iterations=max_iterations_als,
        max_rtpm_iterations=max_iterations_rptm,
        n_initializations=n_initializations).fit(tensor.rclr_transformed)

    # label tensor loadings
    TF.label(tensor, taxonomy=feature_metadata)

    # if the n_components is two add PC3 of zeros
    # this is referenced as in issue in
    # <https://github.com/biocore/emperor/commit
    # /a93f029548c421cb0ba365b4294f7a5a6b0209ce>
    if n_components == 2:
        TF.subjects.loc[:, 'PC3'] = [0] * len(TF.subjects.index)
        TF.features.loc[:, 'PC3'] = [0] * len(TF.features.index)
        TF.proportion_explained['PC3'] = 0
        TF.eigvals['PC3'] = 0

    # save ordination results
    short_method_name = 'CTF_Biplot'
    long_method_name = 'Compositional Tensor Factorization Biplot'
    # only keep PC -- other tools merge metadata
    keep_PC = [col for col in TF.features.columns if 'PC' in col]
    subj_ordin = OrdinationResults(
        short_method_name,
        long_method_name,
        TF.eigvals,
        samples=TF.subjects[keep_PC].dropna(axis=0),
        features=TF.features[keep_PC].dropna(axis=0),
        proportion_explained=TF.proportion_explained)
    # save distance matrix for each condition
    distances = {}
    state_ordn = {}
    subject_trajectories = {}
    feature_trajectories = {}
    for condition, cond, dist, straj, ftraj in zip(tensor.conditions,
                                                   TF.conditions,
                                                   TF.subject_distances,
                                                   TF.subject_trajectory,
                                                   TF.feature_trajectory):
        # match distances to metadata
        ids = straj.index
        ind_dict = dict((ind, ind_i) for ind_i, ind in enumerate(ids))
        inter = set(ind_dict).intersection(sample_metadata.index)
        indices = sorted([ind_dict[ind] for ind in inter])
        dist = dist[indices, :][:, indices]
        distances[condition] = skbio.stats.distance.DistanceMatrix(
            dist, ids=ids[indices])
        # fix conditions
        if n_components == 2:
            cond['PC3'] = [0] * len(cond.index)
        cond = OrdinationResults(short_method_name,
                                 long_method_name,
                                 TF.eigvals,
                                 samples=cond[keep_PC].dropna(axis=0),
                                 features=TF.features[keep_PC].dropna(axis=0),
                                 proportion_explained=TF.proportion_explained)
        state_ordn[condition] = cond
        # add the sample metadata before returning output
        # additionally only keep metadata with trajectory
        # output available.
        pre_merge_cols = list(straj.columns)
        straj = concat([straj.reindex(all_sample_metadata.index),
                        all_sample_metadata],
                       axis=1, sort=True)
        straj = straj.dropna(subset=pre_merge_cols)
        # ensure index name for q2
        straj.index.name = "#SampleID"
        # save traj.
        keep_PC_traj = [col for col in straj.columns
                        if 'PC' in col]
        straj[keep_PC_traj] -= straj[keep_PC_traj].mean()
        ftraj[keep_PC_traj] -= ftraj[keep_PC_traj].mean()
        subject_trajectories[condition] = straj
        ftraj.index = ftraj.index.astype(str)
        feature_trajectories[condition] = ftraj
    return (state_ordn, subj_ordin, distances,
            subject_trajectories, feature_trajectories)
