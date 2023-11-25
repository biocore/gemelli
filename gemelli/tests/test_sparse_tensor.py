import unittest
import pandas as pd
import numpy as np
from biom import Table
from pandas.testing import assert_frame_equal
from gemelli.preprocessing import (build_sparse,
                                   svd_centralize)
from numpy.testing import assert_allclose


class TestBuildSparse(unittest.TestCase):

    def setUp(self):

        # Create a sample table and metadata for testing
        data = np.array([[1,  2,  3,  4,  5],
                         [6,  7,  8,  9, 10],
                         [11, 12, 13, 14, 15]])
        table = Table(data,
                      ['feat1', 'feat2', 'feat3'],
                      ['sample1', 'sample2',
                       'sample3', 'sample4',
                       'sample5'])
        metadata = pd.DataFrame({'sample': ['sample1',
                                            'sample2',
                                            'sample3',
                                            'sample4',
                                            'sample5'],
                                 'individual_id': ['ind1',
                                                   'ind2',
                                                   'ind2',
                                                   'ind1',
                                                   'ind1'],
                                 'state': [1, 2, 1, 2, 1]})
        metadata = metadata.set_index('sample')
        self.table = table
        self.metadata = metadata

    def test_construct_invalid_individual_id(self):
        """
        Test the construct method with invalid ID
        """
        bs = build_sparse()
        with self.assertRaises(ValueError):
            bs.construct(self.table.copy(), self.metadata.copy(),
                         'invalid_id', 'state')

    def test_construct_invalid_state_column(self):
        """
        Test the construct method with invalid states
        """
        bs = build_sparse()
        with self.assertRaises(ValueError):
            bs.construct(self.table.copy(), self.metadata.copy(),
                         'individual_id', 'invalid_state')

    def test_construct_invalid_table(self):
        """
        Test the construct method with invalid table
        """
        bs = build_sparse()
        with self.assertRaises(ValueError):
            bs.construct('invalid_table', self.metadata.copy(),
                         'individual_id', 'state')

    def test_construct_replicate_handling_error(self):
        """
        Test the construct method with replicate_handling='error'
        """
        bs = build_sparse()
        with self.assertRaises(ValueError):
            bs.construct(self.table.copy(), self.metadata.copy(),
                         'individual_id', 'state',
                         replicate_handling='error')

    def test_construct_with_dataframe_table(self):
        """
        Test the construct method works with dataframes
        """
        bs = build_sparse()
        table_df = pd.DataFrame(self.table.matrix_data.toarray(),
                                self.table.ids('observation'),
                                self.table.ids('sample'))
        bs.construct(table_df,
                     self.metadata.copy(),
                     'individual_id', 'state')
        # Check if the constructed attributes are set correctly
        self.assertTrue(isinstance(bs.table, pd.DataFrame))

    def test_construct_replicate_handling_random(self):
        """
        Test the construct method with replicate_handling='random'
        """

        # epxected result
        table_dereplicated_exp = pd.DataFrame(np.array([[1.,  4.,  2.,  3.],
                                                        [6.,  9.,  7.,  8.],
                                                        [11., 14., 12., 13.]]),
                                              ['feat1', 'feat2', 'feat3'],
                                              ['sample1', 'sample4',
                                               'sample2', 'sample3'],)
        mf_dereplicated_exp = pd.DataFrame(np.array([['ind1', 1],
                                                     ['ind1', 2],
                                                     ['ind2', 2],
                                                     ['ind2', 1]]),
                                           ['sample1', 'sample4',
                                            'sample2', 'sample3'],
                                           ['individual_id', 'state'])
        mf_dereplicated_exp['state'] = mf_dereplicated_exp['state'].astype(int)
        t1 = np.array([[1., 4.], [6., 9.], [11., 14.]])
        t2 = np.array([[3., 2.], [8., 7.], [13., 12.]])
        individual_id_tables_exp = {'ind1': pd.DataFrame(t1,
                                                         ['feat1',
                                                          'feat2',
                                                          'feat3'],
                                                         ['sample1',
                                                          'sample4']),
                                    'ind2': pd.DataFrame(t2,
                                                         ['feat1',
                                                          'feat2',
                                                          'feat3'],
                                                         ['sample3',
                                                          'sample2'])}
        individual_id_tables_exp['ind1'].index.name = None
        individual_id_tables_exp['ind2'].index.name = None

        # run and test
        bs = build_sparse()
        bs.construct(self.table.copy(),
                     self.metadata.copy(),
                     'individual_id', 'state',
                     replicate_handling='random',
                     transformation=lambda x: x,
                     pseudo_count=0)
        # test dataframes are the same
        assert_frame_equal(table_dereplicated_exp, bs.table_dereplicated)
        bs.mf_dereplicated.index.name = None
        assert_frame_equal(mf_dereplicated_exp, bs.mf_dereplicated)
        bs.individual_id_tables['ind1'].columns.name = None
        bs.individual_id_tables['ind2'].columns.name = None
        bs.individual_id_tables['ind1'].index.name = None
        bs.individual_id_tables['ind2'].index.name = None
        assert_frame_equal(individual_id_tables_exp['ind1'],
                           bs.individual_id_tables['ind1'])
        assert_frame_equal(individual_id_tables_exp['ind2'],
                           bs.individual_id_tables['ind2'])

    def test_construct_replicate_handling_sum(self):
        """
        Test the construct method with replicate_handling='sum'
        """

        # epxected result
        table_dereplicated_exp = pd.DataFrame(np.array([[6.,  4.,  2.,  3.],
                                                        [16.,  9.,  7.,  8.],
                                                        [26., 14., 12., 13.]]),
                                              ['feat1', 'feat2', 'feat3'],
                                              ['sample1', 'sample4',
                                               'sample2', 'sample3'],)
        mf_dereplicated_exp = pd.DataFrame(np.array([['ind1', 1],
                                                     ['ind1', 2],
                                                     ['ind2', 2],
                                                     ['ind2', 1]]),
                                           ['sample1', 'sample4',
                                            'sample2', 'sample3'],
                                           ['individual_id', 'state'])

        mf_dereplicated_exp['state'] = mf_dereplicated_exp['state'].astype(int)
        t1 = np.array([[6.,  4.], [16.,  9.], [26., 14.]])
        t2 = np.array([[3.,  2.], [8.,  7.], [13., 12.]])
        individual_id_tables_exp = {'ind1': pd.DataFrame(t1,
                                                         ['feat1',
                                                          'feat2',
                                                          'feat3'],
                                                         ['sample1',
                                                          'sample4']),
                                    'ind2': pd.DataFrame(t2,
                                                         ['feat1',
                                                          'feat2',
                                                          'feat3'],
                                                         ['sample3',
                                                          'sample2'])}

        # run and test
        bs = build_sparse()
        bs.construct(self.table.copy(),
                     self.metadata.copy(),
                     'individual_id', 'state',
                     replicate_handling='sum',
                     transformation=lambda x: x,
                     pseudo_count=0)
        # test dataframes are the same
        assert_frame_equal(table_dereplicated_exp, bs.table_dereplicated)
        bs.mf_dereplicated.index.name = None
        assert_frame_equal(mf_dereplicated_exp, bs.mf_dereplicated)
        bs.individual_id_tables['ind1'].columns.name = None
        bs.individual_id_tables['ind2'].columns.name = None
        bs.individual_id_tables['ind1'].index.name = None
        bs.individual_id_tables['ind2'].index.name = None
        assert_frame_equal(individual_id_tables_exp['ind1'],
                           bs.individual_id_tables['ind1'])
        assert_frame_equal(individual_id_tables_exp['ind2'],
                           bs.individual_id_tables['ind2'])

    def test_svd_centralize(self):
        """
        test svd_centralize
        """

        # Create a list of dataframes
        df1 = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 4, 6, 8, 10]})
        df2 = pd.DataFrame({'x': [6, 7, 8, 9, 10], 'y': [12, 14, 16, 18, 20]})
        individual_id_tables = {'subject_one': df1, 'subject_two': df2}

        # Test the svd_centralize function
        results, _, _, _ = svd_centralize(individual_id_tables)

        # Check the results
        exp_one = np.array([[-2.29656623, -1.29656623],
                            [-2.01683978, -0.01683978],
                            [-1.73711332,  1.26288668],
                            [-1.45738687,  2.54261313],
                            [-1.17766041,  3.82233959]])
        exp_two = np.array([[-2.28516848,  3.71483152],
                            [-3.09541201,  3.90458799],
                            [-3.90565554,  4.09434446],
                            [-4.71589906,  4.28410094],
                            [-5.52614259,  4.47385741]])
        assert_allclose(results['subject_one'].values, exp_one, atol=1e-3)
        assert_allclose(results['subject_two'].values, exp_two, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
