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
from typing import Union
from skbio import TreeNode, OrdinationResults, DistanceMatrix
from gemelli.matrix_completion import MatrixCompletion
from gemelli.preprocessing import (matrix_rclr,
                                   fast_unifrac,
                                   bp_read_phylogeny,
                                   retrieve_phylogeny,
                                   create_taxonomy_metadata)
from gemelli._defaults import (DEFAULT_COMP, DEFAULT_MTD,
                               DEFAULT_MSC, DEFAULT_MFC,
                               DEFAULT_OPTSPACE_ITERATIONS,
                               DEFAULT_MFF)
from scipy.linalg import svd
from q2_types.tree import NewickFormat


def phylogenetic_rpca(table: biom.Table,
                      phylogeny: NewickFormat,
                      taxonomy: pd.DataFrame = None,
                      n_components: Union[int, str] = DEFAULT_COMP,
                      min_sample_count: int = DEFAULT_MSC,
                      min_feature_count: int = DEFAULT_MFC,
                      min_feature_frequency: float = DEFAULT_MFF,
                      min_depth: int = DEFAULT_MTD,
                      max_iterations: int = DEFAULT_OPTSPACE_ITERATIONS) -> (
                          OrdinationResults, DistanceMatrix,
                          TreeNode, biom.Table, pd.DataFrame):
    """Runs phylogenetic RPCA.

       This code will be run by both the standalone and QIIME 2 versions of
       gemelli.
    """
    
    # use helper to process table
    table = rpca_table_processing(table,
                                  min_sample_count,
                                  min_feature_count,
                                  min_feature_frequency)

    # import the tree based on filtered table and taxonomy
    phylogeny = bp_read_phylogeny(table, phylogeny, min_depth)
    # build the vectorized table
    counts_by_node, tree_index, branch_lengths, fids, otu_ids\
        = fast_unifrac(table, phylogeny)
    # Robust-clt (matrix_rclr) preprocessing
    rclr_table = matrix_rclr(counts_by_node, branch_lengths=branch_lengths)
    # run OptSpace (RPCA)
    ord_res, dist_res = optspace_helper(rclr_table, fids, table.ids())
    # import expanded table
    counts_by_node = biom.Table(counts_by_node.T, fids, table.ids())

    # validate the metadata using q2 as a wrapper
    if taxonomy is not None and not isinstance(taxonomy, pd.DataFrame):
        taxonomy = taxonomy.to_dataframe()

    # collect taxonomic information for all tree nodes
    postorder_taxonomy = retrieve_phylogeny(phylogeny, taxonomy)
    taxonomy = create_taxonomy_metadata(phylogeny, postorder_taxonomy)

    return ord_res, dist_res, phylogeny, counts_by_node, taxonomy


def rpca(table: biom.Table,
         n_components: Union[int, str] = DEFAULT_COMP,
         min_sample_count: int = DEFAULT_MSC,
         min_feature_count: int = DEFAULT_MFC,
         min_feature_frequency: float = DEFAULT_MFF,
         max_iterations: int = DEFAULT_OPTSPACE_ITERATIONS) -> (
        OrdinationResults,
        DistanceMatrix):
    """Runs RPCA with an matrix_rclr preprocessing step.

       This code will be run by both the standalone and QIIME 2 versions of
       gemelli.
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
                                        table.ids())

    return ord_res, dist_res


def optspace_helper(rclr_table: np.array,
                    feature_ids: list,
                    subject_ids: list,
                    n_components: Union[int, str] = DEFAULT_COMP,
                    max_iterations: int = DEFAULT_OPTSPACE_ITERATIONS) -> (
                        OrdinationResults,
                        DistanceMatrix):

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
        raise ValueError('Data-table contains duplicate indices')
    if len(table.ids('observation')) != len(set(table.ids('observation'))):
        raise ValueError('Data-table contains duplicate columns')

    return table
