import unittest
import numpy as np
from scipy.linalg import qr
from gemelli.factorization import TensorFactorization, khatri_rao


class TestTenAls(unittest.TestCase):

    def setUp(self):
        # generate random noiseless low-rank orthogonal tensor
        r = 3  # rank is 2
        self.r = r
        n1 = 10
        n2 = 12
        n3 = 8
        n4 = 6
        n5 = 2
        U01 = np.random.rand(n1, r)
        U02 = np.random.rand(n2, r)
        U03 = np.random.rand(n3, r)
        U04 = np.random.rand(n4, r)
        U05 = np.random.rand(n5, r)

        # QR factorization ensures component factors are orthogonal
        U1, temp = qr(U01)
        U2, temp = qr(U02)
        U3, temp = qr(U03)
        U4, temp = qr(U04)
        U5 = U05
        U1 = U1[:, 0:r]
        U2 = U2[:, 0:r]
        U3 = U3[:, 0:r]
        U4 = U4[:, 0:r]
        U5 = U5[:, 0:r]
        T = np.zeros((n1, n2, n3))
        for i in range(n3):
            T[:, :, i] = np.matmul(U1, np.matmul(np.diag(U3[i, :]), U2.T))
        to_multiply = [U5, U5, U4, U5]
        to_multiply5 = [U5, U5, U4, U5, U5]
        product = khatri_rao(to_multiply)
        product5 = khatri_rao(to_multiply5)
        T4 = product.sum(1).reshape((n5, n5, n4, n5))
        T5 = product5.sum(1).reshape((n5, n5, n4, n5, n5))
        # sample entries
        p = 2 * (r ** 0.5 * np.log(n1 * n2 * n3)) / np.sqrt(n1 * n2 * n3)
        p4 = 1 * (r ** 5e-5 * np.log(n5 * n5 * n4 * n5)) / np.sqrt(n5 * n5 * n4
                                                                   * n5)
        p5 = 1 * (r ** 5e-5 * np.log(n5 * n5 * n4 * n5 * n5)) / \
            np.sqrt(n5 * n5 * n4 * n5 * n5)
        self.E = abs(np.ceil(np.random.rand(n1, n2, n3) - 1 + p))
        self.E4 = abs(np.ceil(np.random.rand(n5, n5, n4, n5) - 1 + p4))
        self.E5 = abs(np.ceil(np.random.rand(n5, n5, n4, n5, n5) - 1 + p5))
        self.TE = T * self.E
        self.TE4 = T4 * self.E4
        self.TE5 = T5 * self.E5
        self.TE_noise = self.TE + (0.0001 / np.sqrt(n1 * n2 * n3)
                                   * np.random.randn(n1, n2, n3) * self.E)
        self.TE_noise4 = self.TE4 + (1e-10 / np.sqrt(n5 * n5 * n4 * n5)
                                     * np.random.randn(n5, n5,
                                                       n4, n5) * self.E4)
        self.TE_noise5 = self.TE5 + (1e-10 / np.sqrt(n5 * n5 * n4 * n5 * n5)
                                     * np.random.randn(n5, n5, n4, n5, n5
                                                       ) * self.E5)
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n4
        self.n5 = n5
        self.U1 = U1
        self.U2 = U2
        self.U3 = U3
        self.U4 = U4
        self.U5 = U5

    def test_TenAls_center(self):
        # TensorFactorization no noise w/ centered
        TF = TensorFactorization(center=True).fit(self.TE)
        L1, L2, L3 = TF.loadings
        for l1v in L1.mean(axis=1):
            self.assertLess(l1v, 1)
        for l2v in L2.mean(axis=1):
            self.assertLess(l2v, 1)

    def test_TenAls_noiseless(self):
        # TensorFactorization no noise
        TF = TensorFactorization(center=False).fit(self.TE)
        L1, L2, L3 = TF.loadings
        s = TF.eigvals
        # test accuracy
        rmse = 0
        for i3 in range(self.n3):
            A1 = self.U1
            A2 = np.matmul(self.U2, np.diag(self.U3[i3, :]))
            B1 = L1
            B2 = np.matmul(L2, np.diag(L3[i3, :] * s.T.flatten()))
            rmse += np.trace(np.matmul(np.matmul(A1.T, A1), np.matmul(A2.T,
                                                                      A2))) + \
                np.trace(np.matmul(np.matmul(B1.T, B1), np.matmul(B2.T,
                                                                  B2))) + \
                -2 * np.trace(np.matmul(np.matmul(B1.T, A1), np.matmul(A2.T,
                                                                       B2)))
        self.assertTrue(1e2 > abs(rmse))

    def test_TenAls_mode4_noiseless(self):
        # TODO check values
        TF = TensorFactorization(center=False).fit(self.TE4)
        L1, L2, L3, L4 = TF.loadings
        # test accuracy

    def test_TenAls_mode5_noiseless(self):
        # TODO check values
        TF = TensorFactorization(center=False).fit(self.TE5)
        L1, L2, L3, L4, L5 = TF.loadings

    def test_TenAls_noise(self):
        # TensorFactorization no noise
        TF = TensorFactorization(center=False).fit(self.TE_noise)
        L1, L2, L3 = TF.loadings
        s = TF.eigvals
        # test accuracy
        rmse = 0
        for i3 in range(self.n3):
            A1 = self.U1
            A2 = np.matmul(self.U2, np.diag(self.U3[i3, :]))
            B1 = L1
            B2 = np.matmul(L2, np.diag(L3[i3, :] * s.T.flatten()))
            rmse += np.trace(np.matmul(np.matmul(A1.T, A1), np.matmul(A2.T,
                                                                      A2))) + \
                np.trace(np.matmul(np.matmul(B1.T, B1), np.matmul(B2.T,
                                                                  B2))) + \
                -2 * np.trace(np.matmul(np.matmul(B1.T, A1), np.matmul(A2.T,
                                                                       B2)))
        self.assertTrue(1e2 > abs(rmse))

    def test_TenAls_mode4_noise(self):
        # TODO check values
        TF = TensorFactorization(center=False).fit(self.TE_noise4)
        L1, L2, L3, L4 = TF.loadings

    def test_TenAls_mode5_noise(self):
        # TODO check values
        TF = TensorFactorization(center=False).fit(self.TE_noise5)
        L1, L2, L3, L4, L5 = TF.loadings

    def test_khatri_rao(self):
        multiply_2 = khatri_rao([self.U1, self.U2])
        self.assertEqual((self.n1 * self.n2, self.r), multiply_2.shape)
        multiply_3 = khatri_rao([self.U1, self.U2, self.U3])
        self.assertEqual((self.n1 * self.n2 * self.n3, self.r),
                         multiply_3.shape)
        multiply_4 = khatri_rao([self.U1, self.U2, self.U3, self.U4])
        self.assertEqual((self.n1 * self.n2 * self.n3 * self.n4, self.r),
                         multiply_4.shape)
        multiply_5 = khatri_rao([self.U1, self.U2, self.U3, self.U4, self.U4])
        self.assertEqual((self.n1 * self.n2 * self.n3 * self.n4 * self.n4,
                          self.r),
                         multiply_5.shape)
        multiply_6 = khatri_rao([self.U1, self.U2, self.U3, self.U4,
                                 self.U4, self.U5])
        self.assertEqual((self.n1 * self.n2 * self.n3 * self.n4 * self.n4 *
                          self.n5, self.r),
                         multiply_6.shape)

    def test_errors(self):
        # test not array
        with self.assertRaises(ValueError):
            TensorFactorization(center=False).fit(list(range(10)))
        # test if none missing
        with self.assertRaises(ValueError):
            TensorFactorization(center=False).fit(np.ones((5, 5)))
        # test no nan(s)
        TE_errors = self.TE
        TE_errors[0, :, :] = np.inf
        with self.assertRaises(ValueError):
            TensorFactorization(center=False).fit(TE_errors)
        # test max rank
        with self.assertRaises(ValueError):
            paeram_ = np.max(self.TE_noise.shape) + 10
            TensorFactorization(center=False,
                                n_components=paeram_).fit(self.TE_noise)
