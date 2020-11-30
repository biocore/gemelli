from gemelli.optspace import (
    grassmann_manifold_one,
    cost_function,
    gradient_decent,
    grassmann_manifold_two,
    line_search,
    singular_values,
    OptSpace,
    svd_sort,
    rank_estimate)
import numpy as np
from numpy.linalg import norm
import unittest
import numpy.testing as npt
from scipy.io import loadmat
from skbio.util import get_data_path
from numpy.testing import assert_array_almost_equal


class TestOptspace(unittest.TestCase):
    def setUp(self):
        pass

    def test_rank_estimation(self):
        """Test rank estimation is accurate."""
        N = 100
        D = 5000
        k = 3
        U = np.random.standard_normal(size=(N, k))
        V = np.random.standard_normal(size=(k, D))
        Y = U @ V
        # randomly mask Y
        mask = np.random.random(size=(N, D))
        Y[mask > .5] = 0
        # get eps
        total_nonzeros = np.count_nonzero(Y)
        eps = total_nonzeros / np.sqrt(N * D)
        # estimate
        self.assertEqual(k, rank_estimate(Y, eps))

    def test_G(self):
        """Test first grassmann manifold runs."""
        X = np.ones((10, 10))
        m0 = 2
        r = 2
        exp = grassmann_manifold_one(X, m0, r)
        self.assertAlmostEqual(exp, 0.644944589179)

    def test_G_z_0(self):
        """Test first grassmann manifold converges."""
        X = np.array([[1, 3], [4, 1], [2, 1]])
        m0 = 2
        r = 2
        exp = grassmann_manifold_one(X, m0, r)
        self.assertAlmostEqual(exp, 2.60980232)

    def test_F_t(self):
        """Test cost function coverages."""
        X = np.ones((5, 5))
        Y = np.ones((5, 5))
        E = np.zeros((5, 5))
        E[0, 1] = 1
        E[1, 1] = 1
        S = np.eye(5)
        M_E = np.ones((5, 5)) * 6
        M_E[0, 0] = 1
        m0 = 2
        rho = 0.5
        res = cost_function(X, Y, S, M_E, E, m0, rho)
        exp = 1
        assert_array_almost_equal(res, exp, decimal=3)

    def test_F_t_random(self):
        """Test cost function on random values."""
        # random ones and zeros
        np.random.seed(0)
        X = np.ones((5, 5))
        Y = np.ones((5, 5))
        E = np.random.choice([0, 1], size=(5, 5))
        S = np.eye(5)
        M_E = np.ones((5, 5)) * 6
        M_E[0, 0] = 1
        m0 = 2
        rho = 0.5
        res = cost_function(X, Y, S, M_E, E, m0, rho)
        self.assertAlmostEqual(res, 6.5)

    def test_gradF_t(self):
        """Test gradient decent converges."""
        X = np.ones((5, 5))
        Y = np.ones((5, 5))
        E = np.zeros((5, 5))
        E[0, 1] = 1
        E[1, 1] = 1
        S = np.eye(5)
        M_E = np.ones((5, 5)) * 6
        M_E[0, 0] = 1
        m0 = 2
        rho = 0.5
        res = gradient_decent(X, Y, S, M_E, E, m0, rho)
        exp = np.array([[[1., 1., 1., 1., 1.],
                         [1., 1., 1., 1., 1.],
                         [2., 2., 2., 2., 2.],
                         [2., 2., 2., 2., 2.],
                         [2., 2., 2., 2., 2.]],
                        [[2., 2., 2., 2., 2.],
                         [0., 0., 0., 0., 0.],
                         [2., 2., 2., 2., 2.],
                         [2., 2., 2., 2., 2.],
                         [2., 2., 2., 2., 2.]]])
        npt.assert_allclose(exp, res)

    def test_Gp(self):
        """Test second grassmann manifold converges."""
        X = np.ones((5, 5)) * 3
        X[0, 0] = 2
        m0 = 2
        r = 5
        res = grassmann_manifold_two(X, m0, r)
        exp = np.array(
            [[1.08731273, 1.6309691, 1.6309691, 1.6309691, 1.6309691],
             [3.57804989, 3.57804989, 3.57804989, 3.57804989, 3.57804989],
             [3.57804989, 3.57804989, 3.57804989, 3.57804989, 3.57804989],
             [3.57804989, 3.57804989, 3.57804989, 3.57804989, 3.57804989],
             [3.57804989, 3.57804989, 3.57804989, 3.57804989, 3.57804989]]
        )

        npt.assert_allclose(exp, res)

    def test_getoptT(self):
        """Test gradient decent line search."""
        X = np.ones((5, 5))
        Y = np.ones((5, 5))
        E = np.zeros((5, 5))
        E[0, 1] = 1
        E[1, 1] = 1
        S = np.eye(5)
        M_E = np.ones((5, 5)) * 6
        M_E[0, 0] = 1
        m0 = 2
        rho = 0.5
        W, Z = gradient_decent(X, Y, S, M_E, E, m0, rho)
        res = line_search(X, W, Y, Z, S, M_E, E, m0, rho)
        exp = -9.5367431640625e-08
        npt.assert_allclose(exp, res)

    def test_getoptS_small(self):
        """Test singular values from U and V."""
        data = loadmat(get_data_path('small_test.mat'))

        M_E = np.array(data['M_E'].todense())
        E = data['E']

        x = data['x']
        y = data['y']
        res = singular_values(x, y, M_E, E)
        exp = np.array([[0.93639499, 0.07644197, -0.02828782],
                        [-0.03960841, 0.60787383, 0.00521257],
                        [0.00729038, 0.00785834, 0.67853083]])
        npt.assert_allclose(res, exp, atol=1e-5)

    def test_optspace_original(self):
        """Test OptSpace converges on test dataset."""
        M0 = loadmat(get_data_path('large_test.mat'))['M0']
        M_E = loadmat(get_data_path('large_test.mat'))['M_E']

        M0 = M0.astype(np.float)
        M_E = np.array(M_E.todense()).astype(np.float)
        X, S, Y = OptSpace(n_components=3,
                           max_iterations=11,
                           tol=1e-8).solve(M_E)
        err = X[:, ::-1].dot(S).dot(Y[:, ::-1].T) - M0
        n, m = M0.shape

        res = norm(err, 'fro') / np.sqrt(m * n)
        exp = 0.179
        assert_array_almost_equal(res, exp, decimal=1)

    def test_optspace_ordering(self):
        """Test OptSpace produces reproducible loadings."""
        # the expected sorting
        # for U, S, V.
        s_exp = np.array([[5, 4, 1],
                          [8, 3, 0],
                          [7, 9, 2]])
        U_exp = np.array([[6, 3, 0],
                          [7, 4, 1],
                          [8, 5, 2]])
        V_exp = np.array([[6, 3, 0],
                          [7, 4, 1],
                          [8, 5, 2]])
        # un-sorted U,s,v from SVD.
        s_test = np.array([[5, 1, 4],
                           [7, 2, 9],
                           [8, 0, 3]])
        U_test = np.array([[6, 0, 3],
                           [7, 1, 4],
                           [8, 2, 5]])
        V_test = np.array([[6, 0, 3],
                           [7, 1, 4],
                           [8, 2, 5]])
        # run the sorting in optspace
        U_res, s_res, V_res = svd_sort(U_test,
                                       s_test,
                                       V_test)
        assert_array_almost_equal(U_res, U_exp, decimal=3)
        assert_array_almost_equal(s_res, s_exp, decimal=3)
        assert_array_almost_equal(V_res, V_exp, decimal=3)


if __name__ == "__main__":
    unittest.main()
