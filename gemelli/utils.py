# ----------------------------------------------------------------------------
# Copyright (c) 2019--, gemelli development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import warnings


def match(table, metadata, warn=False):
    """

    Match on dense pandas tables,
    taken from gneiss (now dep.)
    https://github.com/biocore/
    gneiss/blob/master/gneiss/util.py
    """

    subtableids = set(table.index)
    submetadataids = set(metadata.index)
    if len(subtableids) != len(table.index):
        raise ValueError("`table` has duplicate sample ids.")
    if len(submetadataids) != len(metadata.index):
        raise ValueError("`metadata` has duplicate sample ids.")

    idx = subtableids & submetadataids
    if len(idx) == 0:
        raise ValueError(("No more samples left.  Check to make sure that "
                          "the sample names between `metadata` and `table` "
                          "are consistent"))
    if (len(idx) != len(subtableids) or
            len(idx) != len(subtableids)) and warn:
        warnings.warn(str(len(idx) - len(subtableids))
                      + " sample(s) did not match.")

    subtable = table.loc[idx]
    submetadata = metadata.loc[idx]

    return subtable, submetadata
