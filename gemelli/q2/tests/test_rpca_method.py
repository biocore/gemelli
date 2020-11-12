import unittest
from os.path import sep as os_path_sep
import numpy as np
from pandas import read_csv
from biom import Table
from skbio import OrdinationResults
from skbio.stats.distance import DistanceMatrix
from skbio.util import get_data_path
from qiime2 import Artifact
from qiime2.plugins import gemelli as q2gemelli
from gemelli.rpca import rpca, auto_rpca
from gemelli.scripts.__init__ import cli as sdc
from gemelli.simulations import build_block_model
from click.testing import CliRunner
from nose.tools import nottest


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
        CliRunner().invoke(sdc.commands['auto-rpca'],
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
        CliRunner().invoke(sdc.commands['rpca'],
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


if __name__ == "__main__":
    unittest.main()
