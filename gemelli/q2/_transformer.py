import qiime2
import pandas as pd
import numpy as np
from .plugin_setup import plugin
from ._format import TrajectoryFormat


def _read_dataframe(fh):
    # Using `dtype=object` and `set_index` to avoid type casting/inference
    # of any columns or the index.
    df = pd.read_csv(fh, sep='\t', header=0)
    df.set_index(df.columns[0], drop=True, append=False, inplace=True)
    df.index = df.index.astype(str)
    return df


@plugin.register_transformer
def _1(data: pd.DataFrame) -> (TrajectoryFormat):
    ff = TrajectoryFormat()
    with ff.open() as fh:
        data.to_csv(fh, sep='\t', header=True, na_rep=np.nan)
    return ff


@plugin.register_transformer
def _2(ff: TrajectoryFormat) -> (pd.DataFrame):
    with ff.open() as fh:
        return _read_dataframe(fh)


@plugin.register_transformer
def _3(ff: TrajectoryFormat) -> (qiime2.Metadata):
    with ff.open() as fh:
        return qiime2.Metadata(_read_dataframe(fh))
