# ----------------------------------------------------------------------------
# Copyright (c) 2019--, gemelli development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np
from gemelli.optspace import OptSpace
from .base import _BaseImpute
from scipy.spatial import distance


class MatrixCompletion(_BaseImpute):

    def __init__(self, n_components=2, max_iterations=5, tol=1e-5):
        """

        This form of matrix completion uses OptSpace (1). Furthermore,
        here we directly interpret the loadings generated from matrix
        completion as a dimensionality reduction.

        Parameters
        ----------

        X: numpy.ndarray - a rclr preprocessed matrix of shape (M,N)
        N = Features (i.e. OTUs, metabolites)
        M = Samples

        n_components: int, optional : Default is 2
        The underlying rank of the default set
        to 2 as the default to prevent overfitting.

        max_iterations: int, optional : Default is 5
        The number of convex iterations to optimize the solution
        If iteration is not specified, then the default iteration is 5.
        Which redcues to a satisfactory error threshold.

        tol: float, optional : Default is 1e-5
        Error reduction break, if the error reduced is
        less than this value it will return the solution

        Returns
        -------
        U: numpy.ndarray - "Sample Loadings" or the unitary matrix
        having left singular vectors as columns. Of shape (M,n_components)

        s: numpy.ndarray - The singular values,
        sorted in non-increasing order. Of shape (n_components,n_components).

        V: numpy.ndarray - "Feature Loadings" or Unitary matrix
        having right singular vectors as rows. Of shape (N,n_components)

        solution: numpy.ndarray - (U*S*V.transpose()) of shape (M,N)

        distance: numpy.ndarray - Distance between each
        pair of the two collections of inputs. Of shape (M,M)

        Raises
        ------
        ValueError

        ValueError
            `ValueError: n_components must be at least 2`.

        ValueError
            `ValueError: max_iterations must be at least 1`.

        ValueError
            `ValueError: Data-table contains either np.inf or -np.inf`.

        ValueError
            `ValueError: The n_components must be less
             than the minimum shape of the input table`.

        References
        ----------
        .. [1] Keshavan RH, Oh S, Montanari A. 2009. Matrix completion
                from a few entries (2009_ IEEE International
                Symposium on Information Theory

        Examples
        --------
        TODO

        """

        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tol = tol

        return

    def fit(self, X, y=None):
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
        n, m = X_sparse.shape

        # make sure the data is sparse (otherwise why?)
        if np.count_nonzero(np.isinf(X_sparse)) != 0:
            raise ValueError('Contains either np.inf or -np.inf')

        # test n-iter
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        # check the settings for n_components
        if isinstance(self.n_components, str) and \
           self.n_components.lower() == 'auto':
            # estimate the rank of the matrix
            self.n_components = 'auto'
        # check hardset values
        elif isinstance(self.n_components, int):
            if self.n_components > (min(n, m) - 1):
                raise ValueError("n-components must be at most"
                                 " 1 minus the min. shape of the"
                                 " input matrix.")
            if self.n_components < 2:
                raise ValueError("n-components must "
                                 "be at least 2")
        # otherwise rase an error.
        else:
            raise ValueError("n-components must be "
                             "an interger or 'auto'")

        # return solved matrix
        self.U, self.s, self.V = OptSpace(n_components=self.n_components,
                                          max_iterations=self.max_iterations,
                                          tol=self.tol).solve(X_sparse)
        # save the solution (of the imputation)
        self.solution = self.U.dot(self.s).dot(self.V.T)
        self.eigenvalues = np.diag(self.s)
        self.explained_variance_ratio = list(
            self.eigenvalues / self.eigenvalues.sum())
        self.distance = distance.cdist(self.U, self.U)
        self.feature_weights = self.V
        self.sample_weights = self.U

    def fit_transform(self, X, y=None):
        """
        Returns the final SVD of

        U: numpy.ndarray - "Sample Loadings" or the
        unitary matrix having left singular
        vectors as columns. Of shape (M,n_components)

        s: numpy.ndarray - The singular values,
        sorted in non-increasing order. Of shape (n_components,n_components).

        V: numpy.ndarray - "Feature Loadings" or Unitary matrix
        having right singular vectors as rows. Of shape (N,n_components)

        """
        X_sparse = X.copy().astype(np.float64)
        self.X_sparse = X_sparse
        self._fit()
        return self.sample_weights
