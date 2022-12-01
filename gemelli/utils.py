# ----------------------------------------------------------------------------
# Copyright (c) 2019--, gemelli development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import biom
from skbio import OrdinationResults
from gemelli._defaults import DEFAULT_MATCH


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
