import unittest
import numpy as np
import pandas as pd
from biom import Table
from skbio import TreeNode
import numpy.testing as npt
from skbio.stats.composition import clr
from gemelli.preprocessing import (build, tensor_rclr, matrix_closure,
                                   matrix_rclr, fast_unifrac)


class Testpreprocessing(unittest.TestCase):

    def setUp(self):
        # test closeure
        self.close_table_zero = np.array([[2, 2, 6],
                                          [0, 0, 0]])
        self.close_true_zero = np.array([[0.2,  0.2,  0.6],
                                         [np.nan] * 3])
        self.close_table = np.array([[2, 2, 6],
                                     [2, 2, 6]])
        self.close_true = np.array([[0.2,  0.2,  0.6],
                                    [0.2,  0.2,  0.6]])
        # matrix_rclr base tests
        self.cdata1 = np.array([[2, 2, 6],
                                [4, 4, 2]])
        self.cdata2 = np.array([[3, 3, 0],
                                [0, 4, 2]])
        self.true2 = np.array([[0.0, 0.0, np.nan],
                               [np.nan, 0.34657359, -0.34657359]])
        self.bad1 = np.array([1, 2, -1])
        self.bad2 = np.array([1, 2, np.inf])
        self.bad3 = np.array([1, 2, np.nan])
        # test dense
        self.count_data_one = np.array([[2, 2, 6],
                                        [4, 4, 2]])
        # test with zeros
        self.count_data_two = np.array([[3, 3, 0],
                                        [0, 4, 2]])
        # test dense tensor
        self.tensor_true = np.array([[[1, 2, 3],
                                      [4, 5, 6]],
                                     [[7, 8, 9],
                                      [10, 11, 12]],
                                     [[13, 14, 15],
                                      [16, 17, 18]]])
        # make a test tree
        newick = ('((((A:1,B:1)n9:3,C:1)n8:3,(D:1,E:1)n7:2)n4:3'
                  ',((F:1,G:1)n6:4,(H:1)n5:4)n3:3)n1;')
        self.tree = TreeNode.read([newick])
        # make a test tree to raise error
        newick_bad = ('((((A,B)n9,C)n8,(D,E)n7)n4'
                      ',((F,G)n6,(H)n5)n3)n1;')
        self.tree_bad = TreeNode.read([newick_bad])
        # make a test table
        counts = np.array([[2, 0, 3, 3, 1, 2, 3, 4],
                           [2, 1, 4, 1, 0, 1, 4, 0]]).T
        feature_ids = [tip_.name for tip_ in self.tree.tips()]
        subject_ids = ['s1', 's2']
        self.phylo_table = Table(counts, feature_ids, subject_ids)
        # fully labeled tree
        self.ids_true = ['A', 'B', 'n2', 'C',
                         'D', 'E', 'n6', 'n7',
                         'F', 'G', 'H', 'n11', 'n12',
                         'n13', 'n14', 'n15']
        # true lengths
        self.branch_length_true = np.array([1., 1., 3., 1., 1., 1., 3., 2.,
                                           1., 1., 1., 4., 4., 3., 3., 0.])
        # true vector count table
        self.vector_counts_true = np.array([[2,  0,  2,  3,  3,  1,  5,  4,
                                            2,  3,  4,  5,  4,  9,  9, 18],
                                            [2,  1,  3,  4,  1,  0,  7,  1,
                                            1, 4,  0,  5,  0,  8,  5, 13]])
        pass

    def test_closure_missing(self):
        """Test closure with zeros (due to tensor)."""
        # test a case with zeros
        cmat_res = matrix_closure(self.close_table_zero)
        npt.assert_allclose(cmat_res, self.close_true_zero)

    def test_closure(self):
        """Test closure without zeros."""
        # test a case with zeros
        cmat_res = matrix_closure(self.close_table)
        npt.assert_allclose(cmat_res, self.close_true)

    def test_rclr_sparse(self):
        """Test matrix_rclr on sparse data."""
        # test a case with zeros
        cmat = matrix_rclr(self.cdata2)
        npt.assert_allclose(cmat, self.true2)

    def test_rclr_negative_raises(self):
        """Test matrix_rclr ValueError on negative."""
        # test negatives throw value error
        with self.assertRaises(ValueError):
            matrix_rclr(self.bad1)

    def test_rclr_inf_raises(self):
        """Test matrix_rclr ValueError on undefined."""
        # test undefined throw value error
        with self.assertRaises(ValueError):
            matrix_rclr(self.bad2)

    def test_rclr_nan_raises(self):
        """Test matrix_rclr ValueError on missing (as nan)."""
        # test nan throw value error
        with self.assertRaises(ValueError):
            matrix_rclr(self.bad3)

    def test_fast_unifrac_branch_raises(self):
        """Test fast_unifrac ValueError on tree with no branch lengths."""
        # test nan throw value error
        with self.assertRaises(ValueError):
            _ = fast_unifrac(self.phylo_table, self.tree_bad)

    def test_fast_unifrac_maxpost_raises(self):
        """Test fast_unifrac ValueError max_postlevel too large."""
        # test nan throw value error
        with self.assertRaises(ValueError):
            _ = fast_unifrac(self.phylo_table, self.tree, max_postlevel=8)

    def test_fast_unifrac_minsplit_raises(self):
        """Test fast_unifrac ValueError min_splits too large."""
        # test nan throw value error
        with self.assertRaises(ValueError):
            _ = fast_unifrac(self.phylo_table, self.tree, min_splits=8)

    def test_fast_unifrac_mindepth_raises(self):
        """Test fast_unifrac ValueError min_depth too large."""
        # test nan throw value error
        with self.assertRaises(ValueError):
            _ = fast_unifrac(self.phylo_table, self.tree, min_depth=8)

    def test_fast_unifrac(self):
        """Test fast_unifrac table vectorized on tree."""
        # run vectorized table
        tmp_res = fast_unifrac(self.phylo_table, self.tree)
        counts_res, _, branch_lengths_res, fids_res, _ = tmp_res
        # test all expected
        npt.assert_allclose(counts_res,
                            self.vector_counts_true)
        npt.assert_allclose(branch_lengths_res,
                            self.branch_length_true)
        self.assertListEqual(fids_res, self.ids_true)

    def test_build(self):
        """Test building a tensor from metadata (multi-mode) & matrix_rclr."""
        # flatten tensor into matrix
        matrix_counts = self.tensor_true.transpose([0, 2, 1])
        reshape_shape = matrix_counts.shape
        matrix_counts = matrix_counts.reshape(9, 2)
        # build mapping and table dataframe to rebuild
        mapping = np.array([[0, 0, 0, 1, 1, 1, 2, 2, 2],
                            [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        mapping = pd.DataFrame(mapping.T,
                               columns=['ID', 'conditional'])
        table = pd.DataFrame(matrix_counts.T)
        # rebuild the tensor
        tensor = build()
        tensor.construct(table, mapping,
                         'ID', ['conditional'])
        # ensure rebuild tensor is the same as it started
        npt.assert_allclose(tensor.counts,
                            self.tensor_true.astype(float))
        # test tensor is ordered correctly in every dimension
        self.assertListEqual(tensor.subject_order,
                             list(range(3)))
        self.assertListEqual(tensor.feature_order,
                             list(range(2)))
        self.assertListEqual(tensor.condition_orders[0],
                             list(range(3)))
        # test that flattened matrix has the same clr
        # transform as the tensor tensor_rclr
        tensor_clr_true = clr(matrix_counts).reshape(reshape_shape)
        tensor_clr_true = tensor_clr_true.transpose([0, 2, 1])
        npt.assert_allclose(tensor_rclr(tensor.counts),
                            tensor_clr_true)

    def test_errors(self):
        """Test building a tensor error raises."""
        # flatten tensor into matrix
        matrix_counts = self.tensor_true.transpose([0, 2, 1])
        matrix_counts = matrix_counts.reshape(9, 2)
        # build mapping and table dataframe to rebuild
        mapping = np.array([[0, 0, 0, 1, 1, 1, 2, 2, 2],
                            [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        mapping = pd.DataFrame(mapping.T,
                               columns=['ID', 'conditional'])
        table = pd.DataFrame(matrix_counts.T)
        # rebuild the tensor
        tensor = build()
        tensor.construct(table, mapping,
                         'ID', ['conditional'])
        # test less than 2D throws ValueError
        with self.assertRaises(ValueError):
            tensor_rclr(np.array(range(3)))
        # test negatives throws ValueError
        with self.assertRaises(ValueError):
            tensor_rclr(tensor.counts * -1)
        tensor_true_error = self.tensor_true.astype(float)
        tensor_true_error[tensor_true_error <= 10] = np.inf
        # test infs throws ValueError
        with self.assertRaises(ValueError):
            tensor_rclr(tensor_true_error)
        tensor_true_error = self.tensor_true.astype(float)
        tensor_true_error[tensor_true_error <= 10] = np.nan
        # test nan(s) throws ValueError
        with self.assertRaises(ValueError):
            tensor_rclr(tensor_true_error)
        # test matrix_rclr on already made tensor
        with self.assertRaises(ValueError):
            matrix_rclr(self.tensor_true)
        # test matrix_rclr on negatives
        with self.assertRaises(ValueError):
            matrix_rclr(self.tensor_true * -1)
        # test that missing id in mapping ValueError
        with self.assertRaises(ValueError):
            tensor.construct(table, mapping.drop(['ID'], axis=1),
                             'ID', ['conditional'])
        # test that missing conditional in mapping ValueError
        with self.assertRaises(ValueError):
            tensor.construct(table, mapping.drop(['conditional'], axis=1),
                             'ID', ['conditional'])
        # test negatives throws ValueError
        with self.assertRaises(ValueError):
            tensor.construct(table * -1, mapping,
                             'ID', ['conditional'])
        table_error = table.astype(float)
        table_error[table_error <= 10] = np.inf
        # test infs throws ValueError
        with self.assertRaises(ValueError):
            tensor.construct(table_error, mapping,
                             'ID', ['conditional'])
        table_error = table.astype(float)
        table_error[table_error <= 10] = np.nan
        # test nan(s) throws ValueError
        with self.assertRaises(ValueError):
            tensor.construct(table_error, mapping,
                             'ID', ['conditional'])
        # test adding up counts for repeat samples
        table[9] = table[8] - 1
        mapping.loc[9, ['ID', 'conditional']
                    ] = mapping.loc[8, ['ID', 'conditional']]
        with self.assertWarns(Warning):
            tensor.construct(table, mapping, 'ID', ['conditional'])
        duplicate_tensor_true = self.tensor_true.copy()
        duplicate_tensor_true[2, :, 2] = duplicate_tensor_true[2, :, 2] - 1
        npt.assert_allclose(tensor.counts,
                            duplicate_tensor_true.astype(float))

    def test_matrix_tensor_rclr(self):
        """Test matrix == tensor matrix_rclr."""
        # test clr works the same if there are no zeros
        npt.assert_allclose(tensor_rclr(self.count_data_one.T).T,
                            clr(self.count_data_one))
        # test a case with zeros
        tensor_rclr(self.count_data_two)
        # test negatives throw ValueError
        with self.assertRaises(ValueError):
            tensor_rclr(self.tensor_true * -1)

    def test_rclr_dense(self):
        """Test matrix_rclr and clr are the same on dense datasets."""
        # test clr works the same if there are no zeros
        cmat = matrix_rclr(self.cdata1)
        npt.assert_allclose(cmat, clr(self.cdata1.copy()))
