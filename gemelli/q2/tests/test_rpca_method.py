import unittest
import numpy as np
import pandas as pd
from pandas import read_csv
from qiime2 import Artifact
from click.testing import CliRunner
from nose.tools import nottest
from biom import Table, load_table
from skbio.util import get_data_path
from os.path import sep as os_path_sep
from gemelli.rpca import rpca, auto_rpca
from gemelli.scripts.__init__ import cli as sdc
from gemelli.simulations import build_block_model
from qiime2.plugins import gemelli as q2gemelli
from skbio import OrdinationResults, TreeNode
from skbio.stats.distance import DistanceMatrix
from numpy.testing import assert_array_almost_equal
from gemelli.testing import assert_ordinationresults_equal


@nottest
def create_test_table():
    _, test_table = build_block_model(rank=2,
                                      hoced=20,
                                      hsced=20,
                                      spar=2e3,
                                      C_=2e3,
                                      num_samples=50,
                                      num_features=500,
                                      mapping_on=False)

    feat_ids = ['F%d' % i for i in range(test_table.shape[0])]
    samp_ids = ['L%d' % i for i in range(test_table.shape[1])]

    return Table(test_table, feat_ids, samp_ids)


class Testrpca(unittest.TestCase):

    def setUp(self):
        self.test_table = create_test_table()

    def test_rpca(self):
        """Tests the basic validity of the actual rpca() method's outputs."""
        ord_test, dist_test = rpca(table=self.test_table)
        # Validate types of the RPCA outputs
        self.assertIsInstance(ord_test, OrdinationResults)
        self.assertIsInstance(dist_test, DistanceMatrix)
        # Ensure that no NaNs are in the OrdinationResults
        # NOTE that we have to use the DataFrame .any() functions instead of
        # python's built-in any() functions -- see #29 for details on this
        self.assertFalse(np.isnan(ord_test.features).any(axis=None))
        self.assertFalse(np.isnan(ord_test.samples).any(axis=None))

    def test_auto_rpca(self):
        """Tests the basic validity of the actual auto_rpca()."""
        ord_test, dist_test = auto_rpca(table=self.test_table)
        # Validate types of the RPCA outputs
        self.assertIsInstance(ord_test, OrdinationResults)
        self.assertIsInstance(dist_test, DistanceMatrix)
        # Ensure that no NaNs are in the OrdinationResults
        # NOTE that we have to use the DataFrame .any() functions instead of
        # python's built-in any() functions -- see #29 for details on this
        self.assertFalse(np.isnan(ord_test.features).any(axis=None))
        self.assertFalse(np.isnan(ord_test.samples).any(axis=None))


class Test_qiime2_rpca(unittest.TestCase):

    def setUp(self):
        self.q2table = Artifact.import_data("FeatureTable[Frequency]",
                                            create_test_table())

    def test_qiime2_auto_rpca(self):
        """ Test Q2 rank estimate matches standalone."""

        tstdir = "test_output"
        # Run gemelli through QIIME 2 (specifically, the Artifact API)
        res = q2gemelli.actions.auto_rpca(self.q2table)
        ordination_qza, distmatrix_qza = res
        # Get the underlying data from these artifacts
        # q2ordination = ordination_qza.view(OrdinationResults)
        q2distmatrix = distmatrix_qza.view(DistanceMatrix)

        # Next, run gemelli outside of QIIME 2. We're gonna check that
        # everything matches up.
        # ...First, though, we need to write the contents of self.q2table to a
        # BIOM file, so gemelli can understand it.
        self.q2table.export_data(get_data_path("", tstdir))
        q2table_loc = get_data_path('feature-table.biom', tstdir)
        # Derived from a line in test_standalone_rpca()
        tstdir_absolute = os_path_sep.join(q2table_loc.split(os_path_sep)[:-1])

        # Run gemelli outside of QIIME 2...
        result = CliRunner().invoke(sdc.commands['auto-rpca'],
                                    ['--in-biom', q2table_loc,
                                     '--output-dir', tstdir_absolute])
        # ...and read in the resulting output files. This code was derived from
        # test_standalone_rpca() elsewhere in gemelli's codebase.
        # stordination = OrdinationResults.read(get_data_path('ordination.txt',
        #                                                    tstdir))
        stdistmatrix_values = read_csv(
            get_data_path(
                'distance-matrix.tsv',
                tstdir),
            sep='\t',
            index_col=0).values

        # Convert the DistanceMatrix object a numpy array (which we can compare
        # with the other _values numpy arrays we've created from the other
        # distance matrices)
        q2distmatrix_values = q2distmatrix.to_data_frame().values

        # Finaly: actually check the consistency of Q2 and standalone results!
        np.testing.assert_array_almost_equal(q2distmatrix_values,
                                             stdistmatrix_values)
        # check that exit code was 0 (indicating success)
        try:
            self.assertEqual(0, result.exit_code)
        except AssertionError:
            ex = result.exception
            error = Exception('Command failed with non-zero exit code')
            raise error.with_traceback(ex.__traceback__)

    def test_qiime2_rpca(self):
        """Tests that the Q2 and standalone RPCA results match."""

        tstdir = "test_output"
        # Run gemelli through QIIME 2 (specifically, the Artifact API)
        ordination_qza, distmatrix_qza = q2gemelli.actions.rpca(self.q2table)
        # Get the underlying data from these artifacts
        # q2ordination = ordination_qza.view(OrdinationResults)
        q2distmatrix = distmatrix_qza.view(DistanceMatrix)

        # Next, run gemelli outside of QIIME 2. We're gonna check that
        # everything matches up.
        # ...First, though, we need to write the contents of self.q2table to a
        # BIOM file, so gemelli can understand it.
        self.q2table.export_data(get_data_path("", tstdir))
        q2table_loc = get_data_path('feature-table.biom', tstdir)
        # Derived from a line in test_standalone_rpca()
        tstdir_absolute = os_path_sep.join(q2table_loc.split(os_path_sep)[:-1])

        # Run gemelli outside of QIIME 2...
        result = CliRunner().invoke(sdc.commands['rpca'],
                                    ['--in-biom', q2table_loc,
                                     '--output-dir', tstdir_absolute])
        # ...and read in the resulting output files. This code was derived from
        # test_standalone_rpca() elsewhere in gemelli's codebase.
        # stordination = OrdinationResults.read(get_data_path('ordination.txt',
        #                                                    tstdir))
        stdistmatrix_values = read_csv(
            get_data_path(
                'distance-matrix.tsv',
                tstdir),
            sep='\t',
            index_col=0).values

        # Convert the DistanceMatrix object a numpy array (which we can compare
        # with the other _values numpy arrays we've created from the other
        # distance matrices)
        q2distmatrix_values = q2distmatrix.to_data_frame().values

        # Finaly: actually check the consistency of Q2 and standalone results!
        np.testing.assert_array_almost_equal(q2distmatrix_values,
                                             stdistmatrix_values)
        # check that exit code was 0 (indicating success)
        try:
            self.assertEqual(0, result.exit_code)
        except AssertionError:
            ex = result.exception
            error = Exception('Command failed with non-zero exit code')
            raise error.with_traceback(ex.__traceback__)

        # NOTE: This functionality is currently not used due to the inherent
        # randomness in how the test table data is generated (and also because
        # we're already checking the correctness of the standalone gemelli
        # RPCA script), but if desired you can add ground truth data to a data/
        # folder in this directory (i.e. a distance-matrix.tsv and
        # ordination.txt file), and the code below will compare the Q2 results
        # to those files.
        #
        # Read in expected output from data/, similarly to above
        # exordination = OrdinationResults.read(get_data_path(
        #                                       'ordination.txt'))
        # exdistmatrix_values = read_csv(get_data_path('distance-matrix.tsv'),
        #                                sep='\t', index_col=0).values
        #
        # ... And check consistency of Q2 results with the expected results
        # assert_gemelli_ordinationresults_equal(q2ordination, exordination)
        # np.testing.assert_array_almost_equal(q2distmatrix_values,
        #                                      exdistmatrix_values)

    def test_qiime2_phylogenetic_rpca(self):
        """Tests that the Q2 (without taxonomy) & standalone phylogenetic RPCA
           match.
        """

        in_table = get_data_path('test.biom')
        in_tree = get_data_path('tree.nwk')
        out_ = os_path_sep.join(in_table.split(os_path_sep)[:-1])
        runner = CliRunner()
        result = runner.invoke(sdc.commands['phylogenetic-rpca'],
                               ['--in-biom', in_table,
                                '--in-phylogeny', in_tree,
                                '--output-dir', out_])
        # Read the results
        dist_res = pd.read_csv(get_data_path('distance-matrix.tsv'),
                               sep='\t',
                               index_col=0)
        ord_res = OrdinationResults.read(get_data_path('ordination.txt'))
        tree_res = TreeNode.read(get_data_path('labeled-phylogeny.nwk'),
                                 format='newick')
        bt_res = load_table(get_data_path('phylo-table.biom'))

        # Run gemelli through QIIME 2 (specifically, the Artifact API)
        table_test = load_table(in_table)
        q2_table_test = Artifact.import_data("FeatureTable[Frequency]",
                                             table_test)
        tree_test = TreeNode.read(in_tree,
                                  format='newick')
        q2_tree_test = Artifact.import_data("Phylogeny[Rooted]",
                                            tree_test)
        res = q2gemelli.actions.phylogenetic_rpca_without_taxonomy(
            q2_table_test,
            q2_tree_test)
        # biplot, distance, count-tree, count-table
        q2ord, q2dist, q2ctree, q2ctbl = res
        # Get the underlying data from these artifacts
        q2ord = q2ord.view(OrdinationResults)
        q2dist = q2dist.view(DistanceMatrix)
        q2dist = q2dist.to_data_frame()
        q2ctree = q2ctree.view(TreeNode)
        q2ctbl = q2ctbl.view(Table)

        # check table values match
        assert_array_almost_equal(bt_res.matrix_data.toarray(),
                                  q2ctbl.matrix_data.toarray())

        # check renamed names are consistent
        name_check_ = [x.name == y.name for x, y in zip(tree_res.postorder(),
                                                        q2ctree.postorder())]
        name_check_ = all(name_check_)
        self.assertEqual(name_check_, True)

        # Check that the distance matrix matches our expectations
        assert_array_almost_equal(dist_res.values, q2dist.values)

        # Check that the ordination results match our expectations -- checking
        # each value for both features and samples
        assert_ordinationresults_equal(ord_res, q2ord)

        # check that exit code was 0 (indicating success)
        try:
            self.assertEqual(0, result.exit_code)
        except AssertionError:
            ex = result.exception
            error = Exception('Command failed with non-zero exit code')
            raise error.with_traceback(ex.__traceback__)


if __name__ == "__main__":
    unittest.main()
