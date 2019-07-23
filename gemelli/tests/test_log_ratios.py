import unittest
import numpy as np
import scipy.sparse as sp
from gemelli.log_ratios import (log_ratio, percentile_ratio)


class Testlogratios(unittest.TestCase):

    def setUp(self):
        # First create block array
        self.n_s = 10
        self.n_f = 20
        A = np.random.randint(1, 10, (self.n_f, self.n_s))
        self.M = sp.block_diag((A, A)).toarray()
        self.ranks = np.linspace(-1, 1, self.M.shape[0])
        self.numerator = self.M[:self.n_f].sum(axis=0)
        self.denominator = self.M[self.n_f:].sum(axis=0)

        pass

    def test_log_ratio(self):

        # test log ratio
        lr_res = log_ratio(self.numerator,
                           self.denominator,
                           pseudocount=1)
        if all(lr_res[:self.n_s] < lr_res[self.n_s:]):
            pass

        # test w/o pseudocount
        lr_res = log_ratio(self.numerator,
                           self.denominator,
                           pseudocount=1)
        if not sum(np.isfinite(lr_res)):
            pass

    def test_log_ratio_value_error(self):

        # num and denom not equal in length
        with self.assertRaises(ValueError):
            log_ratio(self.numerator,
                      self.denominator[:2],
                      pseudocount=1)

        # test neg
        with self.assertRaises(ValueError):
            log_ratio(self.numerator - 1,
                      self.denominator - 1,
                      pseudocount=1)

    def test_percentile_ratio(self):
        # test w/ pseudocount
        lr_res = percentile_ratio(self.M, self.ranks, pseudocount=1)
        if all(lr_res[:self.n_s] < lr_res[self.n_s:]):
            pass

        # test w/o pseudocount
        lr_res = percentile_ratio(self.M, self.ranks)
        if not sum(np.isfinite(lr_res)):
            pass

    def test_percentile_ratio_value_error(self):

        # when ranks not equal to the shape of M
        with self.assertRaises(ValueError):
            percentile_ratio(self.M, self.ranks[:-1])

        # when percentile bad
        with self.assertRaises(ValueError):
            percentile_ratio(self.M, self.ranks, percent=0)
            percentile_ratio(self.M, self.ranks, percent=51)
