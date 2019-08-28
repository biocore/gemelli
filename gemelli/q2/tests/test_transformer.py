import pandas as pd
import qiime2
import numpy as np
import pandas.util.testing as pdt
from skbio.util import get_data_path
from qiime2.plugin.testing import TestPluginBase
from gemelli.q2._format import TrajectoryFormat


class TestTrajectoryFormatTransformers(TestPluginBase):
    package = "gemelli.q2.tests"

    def test_pd_series_to_first_differences_format(self):

        transformer = self.get_transformer(pd.DataFrame, TrajectoryFormat)
        test_df = pd.DataFrame(np.random.normal(size=(10, 3)),
                               [str(i) for i in range(10)],
                               ['PC1', 'PC2', 'PC3'])
        test_df.index.name = '#SampleID'
        result = transformer(test_df)
        self.assertIsInstance(result, TrajectoryFormat)

    def test_trajectory_format_to_pd_dataframe(self):
        _, obs = self.transform_format(TrajectoryFormat,
                                       pd.DataFrame,
                                       'trajectory.tsv')
        exp = pd.read_csv(get_data_path('trajectory.tsv'),
                          sep='\t',
                          index_col=0)
        exp.index = exp.index.astype(str)
        exp.columns = exp.columns.astype(str)
        pdt.assert_frame_equal(obs, exp)

    def test_trajectory_format_to_metadata(self):
        _, obs = self.transform_format(TrajectoryFormat,
                                       qiime2.Metadata,
                                       'trajectory.tsv')
        exp = pd.read_csv(get_data_path('trajectory.tsv'),
                          sep='\t',
                          index_col=0)
        exp.index = exp.index.astype(str)
        exp.columns = exp.columns.astype(str)
        exp = qiime2.Metadata(exp)
        self.assertEqual(obs, exp)
