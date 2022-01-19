import unittest
import pandas as pd
from skbio import TreeNode
from biom import load_table
from os.path import sep as os_path_sep
from click.testing import CliRunner
from skbio.util import get_data_path
from numpy.testing import (assert_allclose,
                           assert_array_almost_equal)
from gemelli.testing import (CliTestCase,
                             absolute_sort)
from gemelli.scripts._standalone_ctf import (standalone_ctf,
                                             standalone_phylogenetic_ctf)


class Test_standalone_ctf(unittest.TestCase):
    def setUp(self):
        pass

    def test_standalone_ctf(self):
        """Checks the output produced by gemelli's standalone CTF script.

           This is more of an "integration test" than a unit test -- the
           details of the algorithm used by the standalone CTF script are
           checked in more detail in gemelli/tests/test_factorization.py.
        """
        in_table = get_data_path('test-small.biom')
        in_meta = get_data_path('test-small.tsv')
        out_ = os_path_sep.join(in_table.split(os_path_sep)[:-1])
        runner = CliRunner()
        result = runner.invoke(standalone_ctf,
                               ['--in-biom',
                                in_table,
                                '--sample-metadata-file',
                                in_meta,
                                '--individual-id-column',
                                'host_subject_id',
                                '--state-column-1',
                                'context',
                                '--output-dir',
                                out_])
        # check exit code was 0 (indicating success)
        CliTestCase().assertExitCode(0, result)
        # Read the results
        samp_res = pd.read_csv(
            get_data_path('context-subject-ordination.tsv'),
            sep='\t',
            index_col=0)
        feat_res = pd.read_csv(
            get_data_path('context-features-ordination.tsv'),
            sep='\t',
            index_col=0)
        # Read the expected results
        samp_exp = pd.read_csv(
            get_data_path('expected-context-subject-ordination.tsv'),
            sep='\t',
            index_col=0)
        feat_exp = pd.read_csv(
            get_data_path('expected-context-features-ordination.tsv'),
            sep='\t',
            index_col=0)
        # Check that the distance matrix matches our expectations
        comp_col = ['PC1', 'PC2', 'PC3']
        cent_ = samp_res[comp_col].mean().values.max()
        self.assertAlmostEqual(cent_, 0)
        cent_ = feat_res[comp_col].mean().values.max()
        self.assertAlmostEqual(cent_, 0)
        # check matched
        assert_allclose(absolute_sort(samp_res[comp_col].values),
                        absolute_sort(samp_exp[comp_col].values),
                        atol=.5)
        assert_allclose(absolute_sort(feat_res[comp_col].values),
                        absolute_sort(feat_exp[comp_col].values),
                        atol=.5)

    def test_standalone_phylogenetic_ctf(self):
        """Checks the output produced by gemelli's standalone phylogenetic CTF.

           This is more of an "integration test" than a unit test -- the
           details of the algorithm used by the standalone CTF script are
           checked in more detail in gemelli/tests/test_factorization.py.
        """
        in_table = get_data_path('test-small.biom')
        in_meta = get_data_path('test-small.tsv')
        in_tree = get_data_path('tree.nwk')
        out_ = os_path_sep.join(in_table.split(os_path_sep)[:-1])
        runner = CliRunner()
        result = runner.invoke(standalone_phylogenetic_ctf,
                               ['--in-biom',
                                in_table,
                                '--in-phylogeny',
                                in_tree,
                                '--sample-metadata-file',
                                in_meta,
                                '--individual-id-column',
                                'host_subject_id',
                                '--state-column-1',
                                'context',
                                '--output-dir',
                                out_])
        # check exit code was 0 (indicating success)
        CliTestCase().assertExitCode(0, result)
        # Read the results
        samp_res = pd.read_csv(
            get_data_path('context-subject-ordination.tsv'),
            sep='\t',
            index_col=0)
        feat_res = pd.read_csv(
            get_data_path('context-features-ordination.tsv'),
            sep='\t',
            index_col=0)
        bt_res = get_data_path('phylogenetic-table.biom')
        bt_res = load_table(bt_res)
        tree_res = get_data_path('expected-labeled-phylogeny.nwk')
        tree_res = TreeNode.read(tree_res,
                                 format='newick')
        # Read the expected results
        samp_exp = pd.read_csv(
            get_data_path('expected-phylo-context-subject-ordination.tsv'),
            sep='\t',
            index_col=0)
        feat_exp = pd.read_csv(
            get_data_path('expected-phylo-context-features-ordination.tsv'),
            sep='\t',
            index_col=0)
        bt_exp = load_table(get_data_path('expected-phylo-table.biom'))
        tree_exp = get_data_path('expected-labeled-phylogeny.nwk')
        tree_exp = TreeNode.read(tree_exp,
                                 format='newick')
        # Check that the distance matrix matches our expectations
        comp_col = ['PC1', 'PC2', 'PC3']
        cent_ = samp_res[comp_col].mean().values.max()
        self.assertAlmostEqual(cent_, 0)
        cent_ = feat_res[comp_col].mean().values.max()
        self.assertAlmostEqual(cent_, 0)
        # check renamed names are consistent
        name_check_ = [x.name == y.name for x, y in zip(tree_res.postorder(),
                                                        tree_exp.postorder())]
        name_check_ = all(name_check_)
        self.assertEqual(name_check_, True)
        # check matched arrays
        assert_allclose(absolute_sort(samp_res[comp_col].values),
                        absolute_sort(samp_exp[comp_col].values),
                        atol=.5)
        assert_allclose(absolute_sort(feat_res[comp_col].values),
                        absolute_sort(feat_exp[comp_col].values),
                        atol=.5)
        assert_array_almost_equal(bt_res.matrix_data.toarray(),
                                  bt_exp.matrix_data.toarray())


if __name__ == "__main__":
    unittest.main()
