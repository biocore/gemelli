import unittest
import os
import numpy as np
from qiime2 import Artifact
import numpy.testing as npt
from biom import load_table, Table
from biom.util import biom_open
from os.path import sep as os_path_sep
from click.testing import CliRunner
from skbio.util import get_data_path
from qiime2.plugins.gemelli.actions import rclr_transformation
from gemelli.scripts.__init__ import cli as sdc


class Test_qiime2_rclr(unittest.TestCase):

    def setUp(self):
        self.cdata = np.array([[3, 3, 0],
                               [0, 4, 2]])
        self.true = np.array([[0.0, 0.0, np.nan],
                              [np.nan, 0.34657359, -0.34657359]])
        pass

    def test__qiime2_rclr(self):
        """Tests q2-rclr matches standalone rclr."""

        # make mock table to write
        samps_ids = ['s%i' % i for i in range(self.cdata.shape[0])]
        feats_ids = ['f%i' % i for i in range(self.cdata.shape[1])]
        table_test = Table(self.cdata.T, feats_ids, samps_ids)
        # write table
        in_ = get_data_path('test.biom', subfolder='data')
        out_path = os_path_sep.join(in_.split(os_path_sep)[:-1])
        test_path = os.path.join(out_path, 'rclr-test.biom')
        with biom_open(test_path, 'w') as wf:
            table_test.to_hdf5(wf, "test")
        # run standalone
        runner = CliRunner()
        result = runner.invoke(sdc.commands['rclr'],
                               ['--in-biom', test_path,
                                '--output-dir', out_path])
        out_table = get_data_path('rclr-table.biom',
                                  subfolder='data')
        res_table = load_table(out_table)
        standalone_mat = res_table.matrix_data.toarray().T
        # check that exit code was 0 (indicating success)
        try:
            self.assertEqual(0, result.exit_code)
        except AssertionError:
            ex = result.exception
            error = Exception('Command failed with non-zero exit code')
            raise error.with_traceback(ex.__traceback__)
        # run QIIME2
        q2_table_test = Artifact.import_data("FeatureTable[Frequency]",
                                             table_test)
        q2_res = rclr_transformation(q2_table_test).rclr_table.view(Table)
        q2_res_mat = q2_res.matrix_data.toarray().T
        # check same and check both correct
        npt.assert_allclose(standalone_mat, q2_res_mat)
        npt.assert_allclose(standalone_mat, self.true)
        npt.assert_allclose(q2_res_mat, self.true)
