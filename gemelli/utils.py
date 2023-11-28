# ----------------------------------------------------------------------------
# Copyright (c) 2019--, gemelli development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import biom
import warnings
import numpy as np
import pandas as pd
from skbio import OrdinationResults, DistanceMatrix
from gemelli._defaults import DEFAULT_MATCH


def qc_distances(distances: DistanceMatrix,
                 table: biom.Table) -> (
                 DistanceMatrix, DistanceMatrix):
    dist_ = distances.to_data_frame()
    # get shared samples
    shared_samples = set(dist_.index) & set(table.ids())
    if len(shared_samples) == 0:
        raise ValueError('No samples overlap between the distances '
                         'and the input table.')
    if len(shared_samples) < len(table.ids()):
        warnings.warn("Less shared samples than samples "
                      "in the table. Could lead to misleading "
                      "results. RPCA distances should be "
                      "rerun with shared sample subset.")
    if len(shared_samples) < len(dist_.index):
        warnings.warn("Less shared samples than samples "
                      "in distances. Could lead to misleading "
                      "results. RPCA distances should be "
                      "rerun with shared sample subset.")
    # match distance
    dist_ = dist_.reindex(shared_samples, axis=1)
    dist_ = dist_.reindex(shared_samples, axis=0)
    dist_ = DistanceMatrix(dist_.values,
                           ids=dist_.index)
    # make the seq. depth diff distance
    samp_sums = pd.Series(table.sum('sample'),
                          table.ids('sample'))
    samp_sum_dist = np.zeros(dist_.shape)
    for i, id_i in enumerate(dist_.ids):
        for j, id_j in enumerate(dist_.ids):
            samp_sum_dist[i][j] = abs(samp_sums[id_i]
                                      - samp_sums[id_j])
    samp_sum_dist = DistanceMatrix(samp_sum_dist,
                                   ids=dist_.ids)
    return dist_, samp_sum_dist


def filter_ordination(ordination: OrdinationResults,
                      table: biom.Table,
                      match_features: bool = DEFAULT_MATCH,
                      match_samples: bool = DEFAULT_MATCH) -> (
                      OrdinationResults):
    """
    This function subsets an OrdinationResults to only those
    samples and features shared with the input table.

    Parameters
    ----------
    ordination: skbio.OrdinationResults
    The biplot ordination in skbio.OrdinationResults format to match.

    table: biom.Table, required
    The feature table in biom format containing the
    samples and features to match with.

    match_features: bool, optional : Default is True
    If set to True the features in the ordination will be matched.

    match_samples: bool, optional : Default is True
    If set to True the samples in the ordination will be matched.

    Returns
    -------
    OrdinationResults
        A matched biplot.

    Raises
    ------
    ValueError
        `ValueError: No features overlap with the input table.`.

    ValueError
        `ValueError: No samples overlap with the input table.`.

    """
    if match_features:
        V = ordination.features
        shared_f = set(V.index) & set(table.ids('observation'))
        if len(shared_f) == 0:
            raise ValueError('No features overlap with the input table.')
        else:
            V = V.reindex(shared_f)
            ordination.features = V  # save subset

    if match_samples:
        U = ordination.samples
        shared_s = set(U.index) & set(table.ids())
        if len(shared_s) == 0:
            raise ValueError('No samples overlap with the input table.')
        else:
            U = U.reindex(shared_s)
            ordination.samples = U  # save subset

    return ordination
