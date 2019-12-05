import unittest
from os.path import sep as os_path_sep
import numpy as np
import pandas as pd
from pandas import read_csv
from biom import load_table
from skbio import OrdinationResults
from skbio.stats.distance import DistanceMatrix
from skbio.util import get_data_path
from qiime2 import Artifact
from qiime2 import Metadata
from qiime2.plugins import gemelli as q2gemelli
from gemelli.ctf import ctf
from gemelli.scripts._standalone_ctf import standalone_ctf
from numpy.testing import assert_allclose
from gemelli.testing import absolute_sort
from click.testing import CliRunner
from nose.tools import nottest


@nottest
def create_test_table():

    in_table = get_data_path('test-small.biom')
    in_meta = get_data_path('test-small.tsv')

    return in_table, in_meta


class Testctf(unittest.TestCase):

    def setUp(self):
        self.in_table, self.in_meta = create_test_table()
        self.subj = 'host_subject_id'
        self.state = 'context'

    def test_ctf(self):
        """Tests the basic validity of the actual ctf() method's outputs."""
        self.biom_table = load_table(self.in_table)
        self.meta_table = read_csv(self.in_meta, sep='\t', index_col=0)
        ord1, ord2, disttst, stst, ftst = ctf(table=self.biom_table,
                                              sample_metadata=self.meta_table,
                                              individual_id_column=self.subj,
                                              state_column=self.state)
        # Validate types of the ctf outputs
        self.assertIsInstance(ord1, OrdinationResults)
        self.assertIsInstance(ord2, OrdinationResults)
        self.assertIsInstance(disttst, DistanceMatrix)
        self.assertIsInstance(stst, pd.DataFrame)
        self.assertIsInstance(ftst, pd.DataFrame)
        # Ensure that no NaNs are in the OrdinationResults
        # NOTE that we have to use the DataFrame .any() functions instead of
        # python's built-in any() functions -- see #29 for details on this
        self.assertFalse(np.isnan(ord1.features).any(axis=None))
        self.assertFalse(np.isnan(ord1.samples).any(axis=None))
        self.assertFalse(np.isnan(ord2.features).any(axis=None))
        self.assertFalse(np.isnan(ord2.samples).any(axis=None))


class Test_qiime2_ctf(unittest.TestCase):

    def setUp(self):
        self.in_table, self.in_meta = create_test_table()
        self.biom_table = load_table(self.in_table)
        self.q2table = Artifact.import_data("FeatureTable[Frequency]",
                                            self.biom_table)
        self.meta_table = read_csv(self.in_meta, sep='\t', index_col=0)
        self.q2meta = Metadata(self.meta_table)
        self.subj = 'host_subject_id'
        self.state = 'context'
        self.out_ = os_path_sep.join(self.in_table.split(os_path_sep)[:-1])

    def test_qiime2_ctf(self):
        """Tests that the Q2 and standalone ctf results match.

           Also validates against ground truth "expected" results.
        """

        # Run gemelli through QIIME 2 (specifically, the Artifact API)
        res = q2gemelli.actions.ctf(table=self.q2table,
                                    sample_metadata=self.q2meta,
                                    individual_id_column=self.subj,
                                    state_column=self.state)
        oqza, osqza, dqza, sqza, fqza = res
        # Get the underlying data from these artifacts
        q2straj = sqza.view(pd.DataFrame)
        q2ftraj = fqza.view(pd.DataFrame)

        # Next, run gemelli outside of QIIME 2. We're gonna check that
        # everything matches up.
        # ...First, though, we need to write the contents of self.q2table to a
        # BIOM file, so gemelli can understand it.
        # Derived from a line in test_standalone_ctf()
        # Run gemelli outside of QIIME 2...
        runner = CliRunner()
        result = runner.invoke(standalone_ctf,
                               ['--in-biom',
                                self.in_table,
                                '--sample-metadata-file',
                                self.in_meta,
                                '--individual-id-column',
                                'host_subject_id',
                                '--state-column-1',
                                'context',
                                '--output-dir',
                                self.out_])
        # check exit code was 0 (indicating success)
        self.assertEqual(result.exit_code, 0)
        # ...and read in the resulting output files. This code was derived from
        # test_standalone_ctf() elsewhere in gemelli's codebase.
        samp_res = read_csv(
            get_data_path('context-subject-ordination.tsv'),
            sep='\t',
            index_col=0)
        feat_res = read_csv(
            get_data_path('context-features-ordination.tsv'),
            sep='\t',
            index_col=0)
        # Check that the trajectory matrix matches our expectations
        comp_col = ['PC1', 'PC2', 'PC3']
        assert_allclose(absolute_sort(samp_res[comp_col].values),
                        absolute_sort(q2straj[comp_col].values),
                        atol=.5)
        assert_allclose(absolute_sort(feat_res[comp_col].values),
                        absolute_sort(q2ftraj[comp_col].values),
                        atol=.5)

    def test_ctf_rank2(self):
        """Tests that ctf with rank < 3
        """
        # Run gemelli
        res = q2gemelli.actions.ctf(table=self.q2table,
                                    sample_metadata=self.q2meta,
                                    individual_id_column=self.subj,
                                    state_column=self.state,
                                    n_components=2)
        # check exit code was 0 (indicating success)
        self.assertEqual(len(res), 5)


if __name__ == "__main__":
    unittest.main()
