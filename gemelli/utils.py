import numpy as np
import pandas as pd
from deicode.preprocessing import rclr

def match(table, metadata):
    """ Match on dense pandas tables,
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
    subtable = table.loc[idx]
    submetadata = metadata.loc[idx]
    return subtable, submetadata
