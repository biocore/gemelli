import unittest
import os
import numpy as np
from skbio import TreeNode
from qiime2 import Artifact
import numpy.testing as npt
from biom import load_table, Table
from biom.util import biom_open
from os.path import sep as os_path_sep
from click.testing import CliRunner
from skbio.util import get_data_path
from numpy.testing import assert_array_almost_equal
from qiime2.plugins.gemelli.actions import (rclr_transformation,
                                            phylogenetic_rclr_transformation)
from gemelli.scripts.__init__ import cli as sdc


class Test_qiime2_rclr(unittest.TestCase):

    def setUp(self):
        self.cdata = np.array([[3, 3, 0],
                               [0, 4, 2]])
        self.true = np.array([[0.0, 0.0, np.nan],
                              [np.nan, 0.34657359, -0.34657359]])
        pass

    def test_qiime2_rclr(self):
        """Tests q2-rclr matches standalone rclr."""

        # make mock table to write
        samps_ids = ['s%i' % i for i in range(self.cdata.shape[0])]
        feats_ids = ['f%i' % i for i in range(self.cdata.shape[1])]
        table_test = Table(self.cdata.T, feats_ids, samps_ids)
        # write table
        in_ = get_data_path('test-smallest.biom',
                            subfolder='data')
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

    def test_qiime2_phylogenetic_rclr(self):
        """Tests q2-phylogenetic-rclr matches standalone rclr."""

        # write table
        in_table = get_data_path('test.biom')
        in_tree = get_data_path('tree.nwk')
        out_path = os_path_sep.join(in_table.split(os_path_sep)[:-1])

        # run phylogenetic-rclr
        runner = CliRunner()
        result = runner.invoke(sdc.commands['phylogenetic-rclr'],
                               ['--in-biom', in_table,
                                '--in-phylogeny', in_tree,
                                '--output-dir', out_path])
        # check that exit code was 0 (indicating success)
        try:
            self.assertEqual(0, result.exit_code)
        except AssertionError:
            ex = result.exception
            error = Exception('Command failed with non-zero exit code')
            raise error.with_traceback(ex.__traceback__)

        # import result
        tree_res = get_data_path('labeled-phylogeny.nwk')
        tree_res = TreeNode.read(tree_res, format='newick')
        counts_res = get_data_path('phylogenetic-count-table.biom')
        counts_res = load_table(counts_res)
        rclr_res = get_data_path('phylogenetic-rclr-table.biom')
        rclr_res = load_table(rclr_res)

        # run QIIME2
        table_test = load_table(in_table)
        q2_table_test = Artifact.import_data("FeatureTable[Frequency]",
                                             table_test)
        tree_test = TreeNode.read(in_tree,
                                  format='newick')
        q2_tree_test = Artifact.import_data("Phylogeny[Rooted]",
                                            tree_test)
        q2_res = phylogenetic_rclr_transformation(q2_table_test,
                                                  q2_tree_test)
        cbn_table, cbn_rclr_table, cbn_tree = q2_res
        q2tree = cbn_tree.view(TreeNode)
        q2cbn_table = cbn_table.view(Table)
        q2cbn_rclr_table = cbn_rclr_table.view(Table)

        # check table values match
        assert_array_almost_equal(q2cbn_table.matrix_data.toarray(),
                                  counts_res.matrix_data.toarray())
        assert_array_almost_equal(q2cbn_rclr_table.matrix_data.toarray(),
                                  rclr_res.matrix_data.toarray())

        # check renamed names are consistent
        name_check_ = [x.name == y.name for x, y in zip(tree_res.postorder(),
                                                        q2tree.postorder())]
        name_check_ = all(name_check_)
        self.assertEqual(name_check_, True)
