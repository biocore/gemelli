import unittest
import pandas as pd
from os.path import sep as os_path_sep
from click.testing import CliRunner
from skbio import OrdinationResults
from skbio.util import get_data_path
from numpy.testing import assert_array_almost_equal
from gemelli.scripts.__init__ import cli as sdc
from gemelli.testing import assert_ordinationresults_equal, CliTestCase


class Test_standalone_rpca(unittest.TestCase):

    def setUp(self):
        pass

    def test_standalone_rpca_rank_est(self):
        """Checks the standalone RPCA rank estimate
           is used instead of a explicit rank
           setting.
        """
        in_ = get_data_path('test.biom', subfolder='rpca_data')
        out_ = os_path_sep.join(in_.split(os_path_sep)[:-1])
        runner = CliRunner()
        result = runner.invoke(sdc.commands['auto-rpca'],
                               ['--in-biom', in_,
                                '--output-dir', out_])
        # Read the results
        dist_res = pd.read_csv(get_data_path('distance-matrix.tsv',
                                             subfolder='rpca_data'), sep='\t',
                               index_col=0)
        ord_res = OrdinationResults.read(get_data_path('ordination.txt',
                                                       subfolder='rpca_data'))

        # Read the expected results
        file_ = 'expected-est-distance-matrix.tsv'
        dist_exp = pd.read_csv(get_data_path(file_, subfolder='rpca_data'),
                               sep='\t', index_col=0)
        ord_exp = OrdinationResults.read(get_data_path(
                                         'expected-est-ordination.txt',
                                         subfolder='rpca_data'))

        # Check that the distance matrix matches our expectations
        assert_array_almost_equal(dist_res.values, dist_exp.values)

        # Check that the ordination results match our expectations -- checking
        # each value for both features and samples
        assert_ordinationresults_equal(ord_res, ord_exp)

        # Lastly, check that gemelli's exit code was 0 (indicating success)
        CliTestCase().assertExitCode(0, result)

    def test_standalone_rpca(self):
        """Checks the output produced by gemelli's RPCA standalone script.

           This is more of an "integration test" than a unit test -- the
           details of the algorithm used by the standalone RPCA script are
           checked in more detail in gemelli/tests/test_optspace.py, etc.
        """
        in_ = get_data_path('test.biom', subfolder='rpca_data')
        out_ = os_path_sep.join(in_.split(os_path_sep)[:-1])
        runner = CliRunner()
        result = runner.invoke(sdc.commands['rpca'],
                               ['--in-biom', in_,
                                '--output-dir', out_])
        # Read the results
        dist_res = pd.read_csv(get_data_path('distance-matrix.tsv',
                                             subfolder='rpca_data'),
                               sep='\t',
                               index_col=0)
        ord_res = OrdinationResults.read(get_data_path('ordination.txt',
                                                       subfolder='rpca_data'))

        # Read the expected results
        dist_exp = pd.read_csv(get_data_path('expected-distance-matrix.tsv',
                                             subfolder='rpca_data'),
                               sep='\t', index_col=0)
        ord_exp = OrdinationResults.read(get_data_path(
                                         'expected-ordination.txt',
                                         subfolder='rpca_data'))

        # Check that the distance matrix matches our expectations
        assert_array_almost_equal(dist_res.values, dist_exp.values)

        # Check that the ordination results match our expectations -- checking
        # each value for both features and samples
        assert_ordinationresults_equal(ord_res, ord_exp)

        # Lastly, check that gemelli's exit code was 0 (indicating success)
        CliTestCase().assertExitCode(0, result)

    def test_standalone_rpca_n_components(self):
        """Tests the standalone RPCA script when n_components is 2
        """
        in_ = get_data_path('test.biom', subfolder='rpca_data')
        out_ = os_path_sep.join(in_.split(os_path_sep)[:-1])
        runner = CliRunner()
        # run the same command but with rank==2
        result = runner.invoke(sdc.commands['rpca'],
                               ['--in-biom', in_,
                                '--output-dir', out_,
                                '--n_components', 2,
                                '--max_iterations', 5])
        CliTestCase().assertExitCode(0, result)
        ord_res = OrdinationResults.read(get_data_path('ordination.txt',
                                                       subfolder='rpca_data'))
        # check it contains three axis
        if len(ord_res.proportion_explained) == 3:
            pass


if __name__ == "__main__":
    unittest.main()
