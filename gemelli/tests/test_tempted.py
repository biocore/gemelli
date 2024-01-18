import unittest
import os
import inspect
import pandas as pd
import numpy as np
from skbio import OrdinationResults
from pandas import read_csv
from biom import load_table
from skbio.util import get_data_path
from gemelli.testing import assert_ordinationresults_equal
from gemelli.tempted import (freg_rkhs,
                             bernoulli_kernel,
                             tempted_transform_helper,
                             tempted_helper,
                             tempted_transform,
                             tempted)
from gemelli.preprocessing import build_sparse
from numpy.testing import assert_allclose


class TestTempted(unittest.TestCase):

    def setUp(self):
        pass

    def test_freg_rkhs(self):
        """
        test freg_rkhs
        """

        Ly = [np.array([-7.511, -13.455, -10.307, -25.813,  26.429]),
              np.array([2.131,  1.225, -3.488, 10.299])]
        a_hat = np.array([0.9, 0.436])
        ind_vec = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
        Kmat = np.array([[1.258, 1.124, 0.995, 0.874,
                          0.758, 1.124, 0.995, 0.874, 0.758],
                        [1.124, 1.064, 1., 0.936,
                         0.874, 1.064, 1., 0.936, 0.874],
                        [0.995, 1., 1.003, 1.,
                         0.995, 1., 1.003, 1., 0.995],
                        [0.874, 0.936, 1., 1.064,
                         1.124, 0.936, 1., 1.064, 1.124],
                        [0.758, 0.874, 0.995, 1.124,
                         1.258, 0.874, 0.995, 1.124, 1.258],
                        [1.124, 1.064, 1., 0.936,
                         0.874, 1.064, 1., 0.936, 0.874],
                        [0.995, 1., 1.003, 1.,
                         0.995, 1., 1.003, 1., 0.995],
                        [0.874, 0.936, 1., 1.064,
                         1.124, 0.936, 1., 1.064, 1.124],
                        [0.758, 0.874, 0.995, 1.124,
                         1.258, 0.874, 0.995, 1.124, 1.258]])
        Kmat_output = np.array([[1.258, 1.124, 0.995, 0.874,
                                 0.758, 1.124, 0.995, 0.874, 0.758],
                                [1.198, 1.098, 0.998, 0.902,
                                 0.809, 1.098, 0.998, 0.902, 0.809],
                                [1.139, 1.071, 1., 0.929,
                                 0.861, 1.071, 1., 0.929, 0.861],
                                [1.08, 1.043, 1.002, 0.958,
                                 0.914, 1.043, 1.002, 0.958, 0.914],
                                [1.023, 1.015, 1.003, 0.986,
                                 0.968, 1.015, 1.003, 0.986, 0.968],
                                [0.968, 0.986, 1.003, 1.015,
                                 1.023, 0.986, 1.003, 1.015, 1.023],
                                [0.914, 0.958, 1.002, 1.043,
                                 1.08, 0.958, 1.002, 1.043, 1.08],
                                [0.861, 0.929, 1., 1.071,
                                 1.139, 0.929, 1., 1.071, 1.139],
                                [0.809, 0.902, 0.998, 1.098,
                                 1.198, 0.902, 0.998, 1.098, 1.198],
                                [0.758, 0.874, 0.995, 1.124,
                                 1.258, 0.874, 0.995, 1.124, 1.258]])
        exp_phi = np.array([-9.275, -16.187,  -2.117,
                            -13.928, -10.251, -25.671,
                            -25.964, -17.503,  -0.885,  36.716])
        res_phi = freg_rkhs(Ly, a_hat, ind_vec, Kmat, Kmat_output)
        assert_allclose(exp_phi, res_phi, atol=1e-3)

    def test_bernoulli_kernel(self):
        """
        test bernoulli_kernel
        """

        Kmat_exp = np.array([[1.25833, 0.99531, 0.75833],
                             [0.99531, 1.00312, 0.99531],
                             [0.75833, 0.99531, 1.25833]])
        Kmat_res = bernoulli_kernel(np.linspace(0, 1, num=3),
                                    np.linspace(0, 1, num=3))
        assert_allclose(Kmat_exp, Kmat_res, atol=1e-3)

    def test_tempted(self):
        """
        Tests tempted and also checks that it matches R version.
        (R v.0.1.0 - 6/1/23)
        """

        callers_filename = inspect.getouterframes(inspect.currentframe())[1][1]
        path = os.path.dirname(os.path.abspath(callers_filename))
        print(path)

        # grab test data
        in_table = get_data_path('test-small.biom', '../q2/tests/data')
        in_meta = get_data_path('test-small.tsv', '../q2/tests/data')
        # get R version expected results
        tempted_rsq_exp = get_data_path('tempted-rsq.csv')
        tempted_rsq_exp = pd.read_csv(tempted_rsq_exp,
                                      index_col=0)
        tempted_fl_exp = get_data_path('tempted-features-loadings.csv')
        tempted_fl_exp = pd.read_csv(tempted_fl_exp,
                                     index_col=0)
        tempted_sl_exp = get_data_path('tempted-state-loadings.csv')
        tempted_sl_exp = pd.read_csv(tempted_sl_exp,
                                     index_col=0)
        tempted_il_exp = get_data_path('tempted-individual-loadings.csv')
        tempted_il_exp = pd.read_csv(tempted_il_exp,
                                     index_col=0)
        tempted_fl_exp.columns = ['component_1', 'component_2', 'component_3']
        tempted_sl_exp.columns = ['component_1', 'component_2', 'component_3']
        tempted_il_exp.columns = ['component_1', 'component_2', 'component_3']
        tempted_il_exp.index = ['subject_6', 'subject_9']
        exp_subject = OrdinationResults('exp', 'exp',
                                        tempted_rsq_exp.values.flatten(),
                                        samples=tempted_il_exp,
                                        features=tempted_fl_exp)
        exp_state = OrdinationResults('exp', 'exp',
                                      tempted_rsq_exp.values.flatten(),
                                      samples=tempted_sl_exp,
                                      features=tempted_fl_exp)
        # run tempted in gemelli
        table = load_table(in_table)
        sample_metadata = read_csv(in_meta, sep='\t', index_col=0)
        sample_metadata['time_points'] = [int(x.split('_')[-1])
                                          for x in sample_metadata['context']]
        # tensor building
        sparse_tensor = build_sparse()
        sparse_tensor.construct(table,
                                sample_metadata,
                                'host_subject_id',
                                'time_points')
        # run TEMPTED
        tbl_cent = sparse_tensor.individual_id_tables_centralized
        tempted_res = tempted_helper(tbl_cent,
                                     sparse_tensor.individual_id_state_orders,
                                     sparse_tensor.feature_order)
        # build res to test
        res_subject = OrdinationResults('exp', 'exp',
                                        tempted_res[4],
                                        samples=tempted_res[0],
                                        features=tempted_res[1])
        tempted_res[2].index = tempted_res[2].index.astype(int) + 1
        res_state = OrdinationResults('exp', 'exp',
                                      tempted_res[4],
                                      samples=tempted_res[2],
                                      features=tempted_res[1])
        # run testing
        assert_ordinationresults_equal(res_subject, exp_subject)
        assert_ordinationresults_equal(res_state, exp_state)

    def test_tempted_wrappers(self):
        """
        Test tempted & tempted projection wrappers.
        """
        # grab test data
        in_table = get_data_path('test-small.biom', '../q2/tests/data')
        in_meta = get_data_path('test-small.tsv', '../q2/tests/data')
        # run tempted in gemelli
        table = load_table(in_table)
        sample_metadata = read_csv(in_meta, sep='\t', index_col=0)
        sample_metadata['time_points'] = [int(x.split('_')[-1])
                                          for x in sample_metadata['context']]
        # run tempted
        ord_res, tdf_, dist_, vdf_ = tempted(table, sample_metadata,
                                             'host_subject_id',
                                             'time_points')
        # project same data as test
        ord_p = tempted_transform(ord_res, tdf_, vdf_,
                                  table,
                                  sample_metadata,
                                  'host_subject_id',
                                  'time_points')
        # make sure the projection is close
        assert_ordinationresults_equal(ord_p, ord_res, precision=0)

    def test_tempted_projection(self):
        """
        Test tempted projection data.
        """
        individual_id_tables_test = {'ID1': pd.DataFrame([[1, 2, 3],
                                                          [4, 5, 6]],
                                                         columns=['Sample1',
                                                                  'Sample2',
                                                                  'Sample3'],
                                                         index=['Feature1',
                                                                'Feature2']),
                                     'ID2': pd.DataFrame([[7, 8, 9],
                                                          [10, 11, 12]],
                                                         columns=['Sample1',
                                                                  'Sample2',
                                                                  'Sample3'],
                                                         index=['Feature1',
                                                                'Feature2'])}
        individual_id_state_orders_test = {'ID1': np.array([1, 2, 3]),
                                           'ID2': np.array([1, 2, 3])}
        feature_loading_train = pd.DataFrame([[0.1, 0.2],
                                              [0.3, 0.4]],
                                             columns=['Component1',
                                                      'Component2'],
                                             index=['Feature1',
                                                    'Feature2'])
        state_loading_train = pd.DataFrame([[0.5, 0.6],
                                            [0.7, 0.8],
                                            [0.9, 1.0]],
                                           columns=['Component1',
                                                    'Component2'],
                                           index=[1, 2, 3])
        eigen_coeff_train = np.array([100, 100])
        time_train = pd.DataFrame([[1], [2], [3]],
                                  index=['Time1',
                                         'Time2',
                                         'Time3'])
        v_centralized_train = np.array([[0.1],
                                        [0.2]])

        # Expected output
        expected_output = pd.DataFrame([[0.022926,
                                         0.025735],
                                        [0.053735,
                                         0.062980]],
                                       columns=['Component1',
                                                'Component2'],
                                       index=['ID1',
                                              'ID2'])

        # Run the function
        output = tempted_transform_helper(individual_id_tables_test,
                                          individual_id_state_orders_test,
                                          feature_loading_train,
                                          state_loading_train,
                                          eigen_coeff_train,
                                          time_train,
                                          v_centralized_train)
        output.round(3).equals(expected_output.round(3))


if __name__ == "__main__":
    unittest.main()
