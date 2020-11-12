import unittest
import numpy as np
from gemelli.matrix_completion import MatrixCompletion
from gemelli.preprocessing import matrix_rclr
from gemelli.optspace import OptSpace
from skbio.stats.composition import clr
from gemelli.q2.tests.simulations import build_block_model
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

    # the matrix_rclr is tested in other places
    # this is just used as input into
    # the OptSpace tests
    test_table = np.array(test_table)
    table_rclr = matrix_rclr(test_table)

    return test_table, table_rclr


class TestOptSpace(unittest.TestCase):

    def setUp(self):
        self.test_table, self.test_rclr = create_test_table()
        self.rank = 2
        self.iteration = 5
        self.tol = 1e-5

    def test_OptSpace(self):
        """Tests the basic validity of the
        actual OptSpace() method's output."""

        # run base OptSpace
        opt = MatrixCompletion(n_components=self.rank,
                               max_iterations=self.iteration,
                               tol=self.tol).fit(self.test_rclr)
        U_res, s_res, V_res = MatrixCompletion(n_components=self.rank,
                                               max_iterations=self.iteration,
                                               tol=self.tol).fit_transform(
                                                   self.test_rclr)
        # use base optspace helper to check
        # that wrapper is not changing outcomes
        U_exp, s_exp, V_exp = OptSpace(n_components=self.rank,
                                       max_iterations=self.iteration,
                                       tol=self.tol).solve(self.test_rclr)
        # more exact testing of directionally is done
        # in test_method.py. Here we just compare abs
        # see  (c/o @cameronmartino's comment in #29).
        for i in range(self.rank):
            np.testing.assert_array_almost_equal(abs(U_exp[:, i]),
                                                 abs(opt.sample_weights[:, i]))
            np.testing.assert_array_almost_equal(abs(s_exp[:, i]),
                                                 abs(opt.s[:, i]))
            np.testing.assert_array_almost_equal(
                abs(V_exp[:, i]), abs(opt.feature_weights[:, i]))
            np.testing.assert_array_almost_equal(abs(U_exp[:, i]),
                                                 abs(U_res[:, i]))
            np.testing.assert_array_almost_equal(abs(s_exp[:, i]),
                                                 abs(s_res[:, i]))
            np.testing.assert_array_almost_equal(abs(V_exp[:, i]),
                                                 abs(V_res[:, i]))

    def test_OptSpace_rank_raises(self):
        """Tests ValueError for OptSpace() rank."""
        # test rank too low
        try:
            MatrixCompletion(n_components=1).fit(self.test_rclr)
        except ValueError:
            pass
        else:
            raise AssertionError("ValueError was not raised")
        # test rank way too high
        try:
            MatrixCompletion(n_components=10000).fit(self.test_rclr)
        except ValueError:
            pass
        else:
            raise AssertionError("ValueError was not raised")
        try:
            MatrixCompletion(n_components=100).fit(self.test_rclr)
        except ValueError:
            pass
        else:
            raise AssertionError("ValueError was not raised")

    def test_OptSpace_iter_raises(self):
        """Tests ValueError for OptSpace() iteration 0."""
        # test iter too low
        try:
            MatrixCompletion(max_iterations=0).fit(self.test_rclr)
        except ValueError:
            pass
        else:
            raise AssertionError("ValueError was not raised")

    def test_OptSpace_illformatted_raises(self):
        """Tests ValueError for OptSpace() no infs."""
        # test inf
        try:
            MatrixCompletion().fit(clr(self.test_table))
        except ValueError:
            pass
        else:
            raise AssertionError("ValueError was not raised")


if __name__ == "__main__":
    unittest.main()
