import unittest
import numpy as np
from scipy.linalg import qr
from gemelli.factorization import tenals


class Testtenals(unittest.TestCase):

    def setUp(self):
        # generate random noiseless low-rank orthogonal tensor
        r = 3  # rank is 2
        n1 = 10
        n2 = 12
        n3 = 8
        U01 = np.random.rand(n1, r)
        U02 = np.random.rand(n2, r)
        U03 = np.random.rand(n3, r)
        U1, temp = qr(U01)
        U2, temp = qr(U02)
        U3, temp = qr(U03)
        U1 = U1[:, 0:r]
        U2 = U2[:, 0:r]
        U3 = U3[:, 0:r]
        T = np.zeros((n1, n2, n3))
        for i in range(n3):
            T[:, :, i] = np.matmul(U1, np.matmul(np.diag(U3[i, :]), U2.T))
        # sample entries
        p = 2 * (r ** 0.5 * np.log(n1 * n2 * n3)) / np.sqrt(n1 * n2 * n3)
        self.E = abs(np.ceil(np.random.rand(n1, n2, n3) - 1 + p))
        self.TE = T * self.E
        self.TE_noise = self.TE + (0.0001 / np.sqrt(n1 * n2 * n3)
                                   * np.random.randn(n1, n2, n3) * self.E)
        self.n3 = n3
        self.U1 = U1
        self.U2 = U2
        self.U3 = U3

    def test_tenals_noiseless(self):
        # TenAls no noise
        loadings, s, dist = tenals(self.TE, self.E)
        L1, L2, L3 = loadings
        s = np.diag(s)
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

    def test_tenals_noise(self):
        # TenAls no noise
        loadings, s, dist = tenals(self.TE_noise, self.E)
        L1, L2, L3 = loadings
        s = np.diag(s)
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
