import numpy as np
from .base import _BaseImpute
from .tenals import  tenals
from scipy.spatial import distance
import warnings


class TenAls(_BaseImpute):

    def __init__(self, rank=3, iteration=50, ninit=50, tol=1e-8):
        """

        TODO

        """

        self.rank = rank
        self.iteration = iteration
        self.ninit = ninit
        self.tol = tol

        return

    def fit(self, X):
        """
        Fit the model to X_sparse
        """

        X_sparse = X.copy().astype(np.float64)
        self.X_sparse = X_sparse
        self._fit()
        return self

    def _fit(self):

        # make copy for imputation, check type
        X_sparse = self.X_sparse

        if not isinstance(X_sparse, np.ndarray):
            X_sparse = np.array(X_sparse)
            if not isinstance(X_sparse, np.ndarray):
                raise ValueError('Input data is should be type numpy.ndarray')
            if len(X_sparse.shape) < 3 or len(X_sparse.shape) > 3:
                raise ValueError('Input data is should be 3rd-order tensor',
                                 ' with shape (samples, features, time)')

        if (np.count_nonzero(X_sparse) == 0 and
                np.count_nonzero(~np.isnan(X_sparse)) == 0):
            raise ValueError('No missing data in the format np.nan or 0')

        if np.count_nonzero(np.isinf(X_sparse)) != 0:
            raise ValueError('Contains either np.inf or -np.inf')

        if self.rank > np.min(X_sparse.shape):
            raise ValueError('rank must be less than the minimum shape')

        
        # return tensor decomp 
        E = np.zeros(X_sparse.shape)
        E[abs(X_sparse)>0] = 1
        U, V, UV_time, s_, dist = tenals(X_sparse,E, r = self.rank, 
                                         ninit = self.ninit, 
                                         nitr = self.iteration, 
                                         tol = self.tol)

        explained_variance_ = (np.diag(s_) ** 2) / (X_sparse.shape[0] - 1)
        ratio = explained_variance_.sum()
        explained_variance_ratio_ = sorted(explained_variance_ / ratio)
        self.eigenvalues = np.diag(s_)
        self.explained_variance_ratio = list(explained_variance_ratio_)[::-1]
        self.distance = distance.cdist(U, U)
        self.time_loading = UV_time
        self.feature_loading = V
        self.sample_loading = U
        self.s = s_

    def fit_transform(self, X):
        """
        TODO

        """
        X_sparse = X.copy().astype(np.float64)
        self.X_sparse = X_sparse
        self._fit()
        return self.sample_loading, self.feature_loading, self.time_loading, self.s
