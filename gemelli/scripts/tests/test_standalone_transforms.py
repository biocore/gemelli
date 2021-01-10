import unittest
import os
import numpy as np
from skbio import TreeNode
import numpy.testing as npt
from biom import load_table, Table
from biom.util import biom_open
from os.path import sep as os_path_sep
from click.testing import CliRunner
from skbio.util import get_data_path
from gemelli.scripts.__init__ import cli as sdc
from gemelli.testing import CliTestCase
from numpy.testing import assert_array_almost_equal


class Test_standalone_rclr(unittest.TestCase):

    def setUp(self):
        self.cdata = np.array([[3, 3, 0],
                               [0, 4, 2]])
        self.true = np.array([[0.0, 0.0, np.nan],
                              [np.nan, 0.34657359, -0.34657359]])
        pass

    def test_standalone_rclr(self):
        """Test the standalone rlcr."""
        # make mock table to write
        samps_ids = ['s%i' % i for i in range(self.cdata.shape[0])]
        feats_ids = ['f%i' % i for i in range(self.cdata.shape[1])]
        table_test = Table(self.cdata.T, feats_ids, samps_ids)

        # write table
        in_ = get_data_path('test.biom', subfolder='rpca_data')
        out_path = os_path_sep.join(in_.split(os_path_sep)[:-1])
        test_path = os.path.join(out_path, 'rclr-test.biom')
        with biom_open(test_path, 'w') as wf:
            table_test.to_hdf5(wf, "test")

        # run rclr command
        runner = CliRunner()
        result = runner.invoke(sdc.commands['rclr'],
                               ['--in-biom', test_path,
                                '--output-dir', out_path])
        out_table = get_data_path('rclr-table.biom',
                                  subfolder='rpca_data')
        res_table = load_table(out_table)
        test_cmat = res_table.matrix_data.toarray().T
        npt.assert_allclose(test_cmat, self.true)
        # Lastly, check that exit code was 0 (indicating success)
        CliTestCase().assertExitCode(0, result)

    def test_standalone_phylogenetic_rclr(self):
        """Test the standalone phylogenetic-rlcr."""

        # write table
        in_table = get_data_path('test.biom', subfolder='rpca_data')
        in_tree = get_data_path('tree.nwk', subfolder='rpca_data')
        out_path = os_path_sep.join(in_table.split(os_path_sep)[:-1])

        # run phylogenetic-rclr
        runner = CliRunner()
        result = runner.invoke(sdc.commands['phylogenetic-rclr'],
                               ['--in-biom', in_table,
                                '--in-phylogeny', in_tree,
                                '--output-dir', out_path])

        # import expected
        tree_exp = get_data_path('expected-labeled-phylogeny.nwk',
                                 subfolder='rpca_data')
        tree_exp = TreeNode.read(tree_exp, format='newick')
        counts_exp = get_data_path('expected-phylogenetic-count-table.biom',
                                   subfolder='rpca_data')
        counts_exp = load_table(counts_exp)
        rclr_exp = get_data_path('expected-phylogenetic-rclr-table.biom',
                                 subfolder='rpca_data')
        rclr_exp = load_table(rclr_exp)

        # import result
        tree_res = get_data_path('labeled-phylogeny.nwk',
                                 subfolder='rpca_data')
        tree_res = TreeNode.read(tree_res, format='newick')
        counts_res = get_data_path('phylogenetic-count-table.biom',
                                   subfolder='rpca_data')
        counts_res = load_table(counts_res)
        rclr_res = get_data_path('phylogenetic-rclr-table.biom',
                                 subfolder='rpca_data')
        rclr_res = load_table(rclr_res)

        # check table values match
        assert_array_almost_equal(counts_exp.matrix_data.toarray(),
                                  counts_res.matrix_data.toarray())
        assert_array_almost_equal(rclr_exp.matrix_data.toarray(),
                                  rclr_res.matrix_data.toarray())

        # check renamed names are consistent
        name_check_ = [x.name == y.name for x, y in zip(tree_res.postorder(),
                                                        tree_exp.postorder())]
        name_check_ = all(name_check_)
        self.assertEqual(name_check_, True)

        # Lastly, check that exit code was 0 (indicating success)
        CliTestCase().assertExitCode(0, result)
