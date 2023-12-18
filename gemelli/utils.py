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
from scipy import stats
from skbio.stats.distance import mantel
from skbio import OrdinationResults, DistanceMatrix
from gemelli._defaults import DEFAULT_MATCH


def qc_rarefaction(table,
                   rarefied_distance,
                   unrarefied_distance,
                   mantel_permutations=1000,
                   return_mantel=False,
                   samp_sum_dist=False):
    """
    Tests if the mantel correlations between
    distances produced from rarefied or from
    unrarefied data differ significantly
    in thier correlation to the absolute
    difference in sample sums.

    This allows for a quick check if you should
    rarefy your data before using RPCA or another
    compositional method that often does not
    require rarefaction.

    If t < 0 and p < alpha then the data should
    be rarefied since the rareified distances
    correlate significantly less to the
    absolute sample sum differences.

    Parameters
    ----------
    table: biom.Table, required
    The _unrarefied_ table from which to calculate
    the abs. difference in total sums.

    rarefied_distance: float, required
    A distance produced with rarefaction.

    unrarefied_distance: float, required
    A distance produced without rarefaction.

    mantel_permutations: int, optional
    Number of permutations for the mantel
    correlation on each comparison.

    return_mantel: bool, optional
    If the mantel results should be returned.

    Returns
    -------
    t and p-value
    If the p-value is less than alpha (often
    set to 0.05) then there is a significant difference
    between the mantel of correlation of the
    absolute sample sum differences and the
    rarefied_distance compared to the
    absolute sample sum differences and the
    unrarefied_distance. So if the t < 0 and the
    p < alpha then the data should be rarefied as
    the rareified distances significantly correlate
    less to the absolute sample sum differences.
    Although it would be unexpected, if t > 0 and
    p < alpha then the unrarefied distance is an
    improvement over the rareified distances.

    """
    # set mantel settings
    method = 'spearman'
    two_tailed = True
    # Find the symmetric difference between ID sets.
    ids1 = set(unrarefied_distance.ids)
    ids2 = set(rarefied_distance.ids)
    ids3 = set(table.ids())
    mismatched_ids = ids1.difference(ids2).difference(ids3)
    mismatched_ids = mismatched_ids.union(ids2.difference(ids1))
    mismatched_ids = mismatched_ids.union(ids3.difference(ids1))
    if mismatched_ids:
        raise ValueError(
            'The following ID(s) are not '
            'contained in both distance matrices '
            'and the input table. This occurs '
            'when mismatched files are passed. '
            'Discard these mismatches before '
            'running this command.\n\n%s'
            % ', '.join(sorted(mismatched_ids)))
    # get the sample sum dist
    if not samp_sum_dist:
        (_, samp_sum_dist) = qc_distances(unrarefied_distance,
                                          table)
    # make sure they are matched
    samp_sum_dist = samp_sum_dist.filter(unrarefied_distance.ids)
    rarefied_distance = rarefied_distance.filter(unrarefied_distance.ids)
    # test if corr(x, y) is diff. than corr(x, z)
    # x = samp_sum_dist
    # y = rarefied_distance
    # z = unrarefied_distance
    (xy,  p_xy, _) = mantel(samp_sum_dist,
                            rarefied_distance,
                            method=method,
                            permutations=mantel_permutations,
                            alternative='two-sided')
    (xz,  pxz_, _) = mantel(samp_sum_dist,
                            unrarefied_distance,
                            method=method,
                            permutations=mantel_permutations,
                            alternative='two-sided')
    (yz,  _, _) = mantel(rarefied_distance,
                         unrarefied_distance,
                         method=method,
                         permutations=mantel_permutations,
                         alternative='two-sided')
    # test results if corr(x, y) is diff. than corr(x, z)
    t_, p_ = steiger(xy, xz, yz,
                     len(rarefied_distance.ids),
                     twotailed=two_tailed)
    # if p_ < alpha, then rarefaction has a signifcantly
    # different correlation to the sample sum diffs
    # than non-rare does to the sample sum diffs
    # (so we want the p_ to be > alpha)
    if not return_mantel:
        return t_, p_
    else:
        return t_, p_, xy, p_xy, xz, pxz_, yz, samp_sum_dist


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


def steiger(xy, xz, yz, n, twotailed=True):
    """
    Steiger method for calculating the statistical
    significant differences between two
    dependent correlation coefficients.

    R package http://personality-project.org/r/html/paired.r.html
    Credit goes to the authors of above mentioned packages!

    Author: Philipp Singer (www.philippsinger.info)

    README.md: CorrelationStats
    This Python script enables you to compute statistical significance
    trests on both dependent and independent correlation coefficients.
    For each case two methods to choose from are available.
    For details, please refer to: http://www.philippsinger.info/?p=347
    #copied from on 11/29/2023 from
    https://github.com/psinger/CorrelationStats/blob/master/corrstats.py

    Parameters
    ----------
    xy: float, required
    correlation coefficient between x and y

    xz: float, required
    correlation coefficient between x and z

    yz: float, required
    correlation coefficient between y and z

    n: int, required
    number of elements in x, y and z

    twotailed: bool, optional : Default is True
    whether to calculate a one or two tailed test

    Returns
    -------
    t and p-val
    """
    d = xy - xz
    determin = 1 - xy * xy - xz * xz - yz * yz + 2 * xy * xz * yz
    av = (xy + xz)/2
    cube = (1 - yz) * (1 - yz) * (1 - yz)

    t2 = d * np.sqrt((n - 1) * (1 + yz) /
                     (((2 * (n - 1)/(n - 3))
                       * determin + av * av * cube)))
    p = 1 - stats.t.cdf(abs(t2), n - 3)

    if twotailed:
        p *= 2

    return t2, p


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
