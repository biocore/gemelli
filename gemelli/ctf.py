import biom
import skbio
import numpy as np
import pandas as pd
from pandas import concat
from pandas import DataFrame
from q2_types.tree import NewickFormat
from skbio import OrdinationResults, DistanceMatrix, TreeNode
from gemelli.factorization import TensorFactorization
from gemelli.preprocessing import (build, tensor_rclr,
                                   fast_unifrac,
                                   bp_read_phylogeny)
from gemelli._defaults import (DEFAULT_COMP, DEFAULT_MSC,
                               DEFAULT_MFC, DEFAULT_BL,
                               DEFAULT_MTD,
                               DEFAULT_TENSALS_MAXITER,
                               DEFAULT_FMETA as DEFFM)


def phylogenetic_ctf(table: biom.Table,
                     phylogeny: NewickFormat,
                     sample_metadata: DataFrame,
                     individual_id_column: str,
                     state_column: str,
                     n_components: int = DEFAULT_COMP,
                     min_sample_count: int = DEFAULT_MSC,
                     min_feature_count: int = DEFAULT_MFC,
                     min_depth: int = DEFAULT_MTD,
                     min_splits: int = DEFAULT_MTD,
                     max_postlevel: int = DEFAULT_MTD,
                     max_iterations_als: int = DEFAULT_TENSALS_MAXITER,
                     max_iterations_rptm: int = DEFAULT_TENSALS_MAXITER,
                     n_initializations: int = DEFAULT_TENSALS_MAXITER,
                     feature_metadata: DataFrame = DEFFM) -> (
                         OrdinationResults, OrdinationResults,
                         DistanceMatrix, DataFrame, DataFrame,
                         TreeNode, biom.Table):

    phylogeny = bp_read_phylogeny(table, phylogeny)
    # run CTF helper and parse output for QIIME
    helper_results = phylogenetic_ctf_helper(table,
                                             phylogeny,
                                             sample_metadata,
                                             individual_id_column,
                                             [state_column],
                                             n_components,
                                             min_sample_count,
                                             min_feature_count,
                                             min_depth,
                                             min_splits,
                                             max_postlevel,
                                             max_iterations_als,
                                             max_iterations_rptm,
                                             n_initializations,
                                             feature_metadata)
    (state_ordn, ord_res, dists, straj,
     ftraj, phylogeny, counts_by_node) = helper_results
    # save only first state (QIIME can't handle a list yet)
    dists = list(dists.values())[0]
    straj = list(straj.values())[0]
    ftraj = list(ftraj.values())[0]
    state_ordn = list(state_ordn.values())[0]

    return ord_res, state_ordn, dists, straj, ftraj, phylogeny, counts_by_node


def phylogenetic_ctf_helper(table: biom.Table,
                            phylogeny: TreeNode,
                            sample_metadata: DataFrame,
                            individual_id_column: str,
                            state_column: list,
                            n_components: int = DEFAULT_COMP,
                            min_sample_count: int = DEFAULT_MSC,
                            min_feature_count: int = DEFAULT_MFC,
                            min_depth: int = DEFAULT_MTD,
                            min_splits: int = DEFAULT_MTD,
                            max_postlevel: int = DEFAULT_MTD,
                            max_iterations_als: int = DEFAULT_TENSALS_MAXITER,
                            max_iterations_rptm: int = DEFAULT_TENSALS_MAXITER,
                            n_initializations: int = DEFAULT_TENSALS_MAXITER,
                            feature_metadata: DataFrame = DEFFM) -> (
                                OrdinationResults, OrdinationResults,
                                DistanceMatrix, DataFrame, DataFrame,
                                TreeNode, biom.Table):

    # check the table for validity and then filter
    process_results = ctf_table_processing(table,
                                           sample_metadata,
                                           individual_id_column,
                                           state_column,
                                           min_sample_count,
                                           min_feature_count,
                                           feature_metadata)
    (table, sample_metadata,
     all_sample_metadata, feature_metadata) = process_results
    # build the vectorized table
    counts_by_node, tree_index, branch_lengths, fids, otu_ids\
        = fast_unifrac(table, phylogeny, min_depth, min_splits, max_postlevel)
    # import expanded table
    counts_by_node = biom.Table(counts_by_node.T,
                                fids, table.ids())
    # add feature index place holders
    # In the future maybe label by internal node clade?
    if feature_metadata is not None:
        # add rows for taxonomy missing (if given)
        new_internal_nodes = set(fids) - set(feature_metadata.index)
        append_taxon = []
        for add_node in new_internal_nodes:
            internal_label_ = ''.join(['k__; ' + add_node,
                                       'p__; ' + add_node,
                                       'c__; ' + add_node,
                                       'o__; ' + add_node,
                                       'f__; ' + add_node,
                                       'g__; ' + add_node,
                                       's__' + add_node])
            confidence_ = 1.0
            append_taxon.append([internal_label_, confidence_])
        append_taxon = pd.DataFrame(append_taxon,
                                    new_internal_nodes,
                                    ['Taxon', 'Confidence'])
        feature_metadata = pd.concat([feature_metadata, append_taxon])
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
                                    feature_metadata)
    state_ordn, ord_res, dists, straj, ftraj = tensal_results

    return state_ordn, ord_res, dists, straj, ftraj, phylogeny, counts_by_node


def ctf(table: biom.Table,
        sample_metadata: DataFrame,
        individual_id_column: str,
        state_column: str,
        n_components: int = DEFAULT_COMP,
        min_sample_count: int = DEFAULT_MSC,
        min_feature_count: int = DEFAULT_MFC,
        max_iterations_als: int = DEFAULT_TENSALS_MAXITER,
        max_iterations_rptm: int = DEFAULT_TENSALS_MAXITER,
        n_initializations: int = DEFAULT_TENSALS_MAXITER,
        feature_metadata: DataFrame = DEFFM) -> (OrdinationResults,
                                                 OrdinationResults,
                                                 DistanceMatrix,
                                                 DataFrame,
                                                 DataFrame):

    # run CTF helper and parse output for QIIME
    helper_results = ctf_helper(table,
                                sample_metadata,
                                individual_id_column,
                                [state_column],
                                n_components,
                                min_sample_count,
                                min_feature_count,
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
               max_iterations_als: int = DEFAULT_TENSALS_MAXITER,
               max_iterations_rptm: int = DEFAULT_TENSALS_MAXITER,
               n_initializations: int = DEFAULT_TENSALS_MAXITER,
               feature_metadata: DataFrame = DEFFM) -> (
                   OrdinationResults, OrdinationResults,
                   DistanceMatrix, DataFrame, DataFrame):
    # check the table for validity and then filter
    process_results = ctf_table_processing(table,
                                           sample_metadata,
                                           individual_id_column,
                                           state_column,
                                           min_sample_count,
                                           min_feature_count,
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


def ctf_table_processing(table: biom.Table,
                         sample_metadata: DataFrame,
                         individual_id_column: str,
                         state_columns: list,
                         min_sample_count: int = DEFAULT_MSC,
                         min_feature_count: int = DEFAULT_MFC,
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

    # filter and import table
    for axis, min_sum in zip(['sample',
                              'observation'],
                             [min_sample_count,
                              min_feature_count]):
        table = table.filter(table.ids(axis)[table.sum(axis) >= min_sum],
                             axis=axis, inplace=True)

    # return data based on input
    if feature_metadata is not None:
        return table, sample_metadata, feature_metadata
    else:
        return table, sample_metadata, all_sample_metadata, None


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
                     state_columns)

    # rclr of slices
    transformed_counts = tensor_rclr(tensor.counts,
                                     branch_lengths=branch_lengths)

    # factorize
    TF = TensorFactorization(
        n_components=n_components,
        max_als_iterations=max_iterations_als,
        max_rtpm_iterations=max_iterations_rptm,
        n_initializations=n_initializations).fit(transformed_counts)

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
