# ----------------------------------------------------------------------------
# Copyright (c) 2019--, gemelli development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np
from numpy.matlib import repmat
from numpy.linalg import norm
from scipy.sparse.linalg import svds


class OptSpace(object):
    """

    OptSpace is a matrix completion algorithm based on a singular value
    decomposition (SVD) optimized on a local manifold. It has been shown to
    be quite robust to noise in low rank datasets (1).
    The objective function that it is trying to optimize
    over is given by:

        min(P|(Y-U*S*V^{T})|_{2}^{2}

    U and V are matrices that are trying to be estimated and S
    is analogous to a matrix of eigenvalues. Y are the
    observed values and P is a function such that
    the errors between Y and USV are only computed
    on the nonzero entries.

    """

    def __init__(
            self,
            n_components,
            max_iterations,
            tol,
            step_size=10000,
            resolution_limit=20,
            sign=-1):
        """

        Parameters
        ----------
        obs: numpy.ndarray - a rclr preprocessed matrix of shape (M,N)
            with missing values set to zero or np.nan.
            N = Features (i.e. OTUs, metabolites)
            M = Samples

        n_components: int or {"optspace"}, optional
            The underlying rank of the dataset.
            This will also control the number of components
            (axis) in the U and V loadings. The value can either
            be hard set by the user or estimated through the
            optspace algorithm.

        max_iterations: int
            The maximum number of convex iterations to optimize the solution
            If iteration is not specified, then the default iteration is 5.
            Which redcues to a satisfactory error threshold.

        tol: float
            Error reduction break, if the error reduced is
            less than this value it will return the solution

        step_size: int, optional : Default is 10000
            The gradient decent step size, this will be
            optimized by the line search.

        resolution_limit: int, optional : Default is 20
            The gradient decent line search resolution limit.
            Where the resolution is line > 2**-resolution_limit.

        sign: int, optional : Default is -1
            Can be one or negative one. This controls the
            sign correction in the gradient decent U, V
            updates.

        Returns
        -------
        self.U: numpy.ndarray - "Sample Loadings" or the unitary matrix
            having left singular vectors as columns.
            Of shape (M, n_components)

        self.s: numpy.ndarray - The singular values,
            sorted in non-increasing order.
            Of shape (n_components, n_components).

        self.V: numpy.ndarray - "Feature Loadings" or Unitary matrix
            having right singular vectors as rows.
            Of shape (N, n_components)

        References
        ----------
        .. [1] Keshavan RH, Oh S, Montanari A. 2009. Matrix completion
                from a few entries (2009_ IEEE International
                Symposium on Information Theory
        """

        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tol = tol
        self.step_size = step_size
        self.resolution_limit = resolution_limit
        self.sign = sign

    def solve(self, obs):
        # adjust iteration indexing by one
        self.max_iterations += 1
        # Convert any nan input to zeros
        # optspace considers zero  and only zero
        # as missing.
        obs[np.isnan(obs)] = 0
        # generate a mask that tracks where missing
        # values exist in the obs dataset
        mask = (np.abs(obs) > 0).astype(int)
        # save the shape of the matrix
        n, m = obs.shape
        # get a measure of sparsity level
        total_nonzeros = np.count_nonzero(mask)
        eps = total_nonzeros / np.sqrt(m * n)
        # raise future warning if hard set
        if isinstance(self.n_components, int):
            if self.n_components > (min(n, m) - 1):
                raise ValueError("n-components must be at most"
                                 " 1 minus the min. shape of the"
                                 " input matrix.")
        # otherwise rase an error.
        else:
            raise ValueError("n-components must be "
                             "an interger.")
        # The rescaling factor compensates the smaller average size of
        # the of the missing entries (mask) with respect to obs
        rescal_param = np.count_nonzero(mask) * self.n_components
        rescal_param = np.sqrt(rescal_param / (norm(obs, 'fro') ** 2))
        obs = obs * rescal_param
        # Our initial first guess are the loadings generated
        # by the traditional SVD
        U, S, V = svds(obs, self.n_components, which='LM')
        # the shape and number of non-zero values
        # can set the input perameters for the gradient
        # decent
        rho = eps * n
        U = U * np.sqrt(n)
        V = (V * np.sqrt(m)).T
        S = S / eps
        # generate the new singular values from
        # the initialization of U and V
        S = singular_values(U, V, obs, mask)
        # initialize the difference between the
        # observed values of the matrix and the
        # the imputed matrix generated by the loadings
        # from this point on we call this "distortion"
        obs_error = obs - U.dot(S).dot(V.T)
        # initialize the distortion matrix of line above
        dist = np.zeros(self.max_iterations + 1)
        # starting initialization of the distortion between obs and imputed
        dist[0] = norm(np.multiply(obs_error, mask), 'fro') / \
            np.sqrt(total_nonzeros)
        # we will perform gradient decent for at most self.max_iterations
        for i in range(1, self.max_iterations):
            # generate new optimized loadings from F(X,Y) [1]
            U_update, V_update = gradient_decent(
                U, V, S, obs, mask, self.step_size, rho)
            # Line search for the optimum jump length in gradient decent
            line = line_search(
                U,
                U_update,
                V,
                V_update,
                S,
                obs,
                mask,
                self.step_size,
                rho,
                resolution_limit=self.resolution_limit)
            # with line and iterations loading update U,V loadings
            U = U - self.sign * line * U_update
            V = V - self.sign * line * V_update
            # generate new singular values from the new
            # loadings
            S = singular_values(U, V, obs, mask)
            # Compute the distortion
            obs_error = obs - U.dot(S).dot(V.T)
            # update the new distortion
            dist[i + 1] = norm(np.multiply(obs_error, mask),
                               'fro') / np.sqrt(total_nonzeros)
            # if the gradient decent has coverged then break the loop
            # and return the results
            if (dist[i + 1] < self.tol):
                break
        # compensates the smaller average size of
        # observed values vs. missing
        S = S / rescal_param
        # ensure the loadings are sorted properly
        U, S, V = svd_sort(U, S, V)
        return U, S, V

    def joint_solve(self, multiple_obs):
        """
        Solver adaptation to OptSpace with
        multiple input tables.

        Parameters
        ----------
        multiple_obs:  list of tuples of numpy.ndarray -
            A list of tuples where the tuples have two matricies
            of shape (M_train, N) and (M_test, N) where
            the M_train/M_test are shared but N is not
            across tables. Missing values set to zero or np.nan.
            N = Features (i.e. OTUs, metabolites)
            M = Samples

        Returns
        -------
        self.U_shared: numpy.ndarray -
            "Sample Loadings" or the unitary matrix
            having left singular vectors as columns.
            Of shape (M, n_components).

        self.S_shared: numpy.ndarray -
            The singular values,
            sorted in non-increasing order.
            Of shape (n_components, n_components).

        self.feature_loadings list of numpy.ndarray -
            "Feature Loadings" or Unitary matrix
            having right singular vectors as rows.
            Of shape (N, n_components). For each
            table in multiple_obs and in the same
            order as  multiple_obs.

        self.dists: numpy.ndarray -
            The cross validation error for the test
            set on the training set.
        """
        # adjust iteration indexing by one
        self.max_iterations += 1
        test_obs = []
        masks = []
        dims = []
        for i_obs, (t_obs, obs) in enumerate(multiple_obs):
            # Convert any nan input to zeros
            # optspace considers zero  and only zero
            # as missing.
            obs[np.isnan(obs)] = 0
            multiple_obs[i_obs] = obs
            # masked test set
            test_obs_m = np.ma.array(t_obs, mask=np.isnan(t_obs))
            test_obs_m = test_obs_m - test_obs_m.mean(axis=1).reshape(-1, 1)
            test_obs_m = test_obs_m - test_obs_m.mean(axis=0)
            test_obs.append(test_obs_m)
            # generate a mask that tracks where missing
            # values exist in the obs dataset
            mask = (np.abs(obs) > 0).astype(int)
            masks.append(mask)
            # save the shape of the matrix
            dims.append(obs.shape)
        # raise future warning if hard set
        if isinstance(self.n_components, int):
            if self.n_components > min([min(n, m) - 1 for n, m in dims]):
                raise ValueError("n-components must be at most"
                                 " 1 minus the min. shape of the"
                                 " smallest input matrix.")
        # otherwise rase an error.
        else:
            raise ValueError("n-components must be "
                             "an interger")

        # new stacked init
        dists = np.zeros((2, self.max_iterations - 1))
        dist_sum_iter = []
        # stack data
        obs_stacked = np.hstack(multiple_obs)
        mask_stacked = np.hstack(masks)
        n, m = obs_stacked.shape
        # get a measure of sparsity level
        total_nonzeros = np.count_nonzero(mask_stacked)
        eps = total_nonzeros / np.sqrt(m * n)
        # The rescaling factor compensates the smaller average size of
        # the of the missing entries (mask) with respect to obs
        rescal_param = np.count_nonzero(mask_stacked) * self.n_components
        rescal_param = np.sqrt(rescal_param / (norm(obs_stacked, 'fro') ** 2))
        obs_stacked = obs_stacked * rescal_param
        # Our initial first guess are the loadings generated
        # by the traditional SVD
        U, S, V = svds(obs_stacked, self.n_components, which='LM')
        # the shape and number of non-zero values
        # can set the input perameters for the gradient
        # decent
        rho = eps * n
        U = U * np.sqrt(n)
        V = (V * np.sqrt(m)).T
        S = S / eps
        # generate the new singular values from
        # the initialization of U and V
        S = singular_values(U, V, obs_stacked, mask_stacked)
        # initialize the difference between the
        # observed values of the matrix and the
        # the imputed matrix generated by the loadings
        # from this point on we call this "distortion"
        obs_error = obs_stacked - U.dot(S).dot(V.T)
        # starting initialization of the distortion between obs and imputed
        # separate feature loadings
        feature_loadings = []
        feat_index_start = 0
        for _, feat_index in dims:
            feat_index_end = feat_index_start + feat_index
            feature_loadings.append(V[feat_index_start:feat_index_end, :])
            feat_index_start += feat_index
        # store shared init
        U_shared = U
        S_shared = S
        # we will perform gradient decent for at most self.max_iterations
        for i in range(1, self.max_iterations):
            # re-init
            dist_sum_iter = []
            sample_loadings = []
            all_singular = []
            for i_obs, V in enumerate(feature_loadings):
                # load saved arrays
                mask = masks[i_obs]
                n, m = dims[i_obs]
                obs = multiple_obs[i_obs]
                # generate new optimized loadings from F(X,Y) [1]
                U_update, V_update = gradient_decent(
                    U_shared, V, S_shared, obs, mask, self.step_size, rho)
                # Line search for the optimum jump length in gradient decent
                line = line_search(
                    U_shared,
                    U_update,
                    V,
                    V_update,
                    S_shared,
                    obs,
                    mask,
                    self.step_size,
                    rho,
                    resolution_limit=self.resolution_limit)
                # with line and iterations loading update U,V loadings
                U = U_shared - self.sign * line * U_update
                V = V - self.sign * line * V_update
                # generate new singular values from the new
                # loadings
                S = singular_values(U_shared, V, obs, mask)
                # Compute the distortion
                obs_error = obs - U.dot(S).dot(V.T)
                # update the new distortion
                # add samples and singular values
                sample_loadings.append(U)
                feature_loadings[i_obs] = V
                all_singular.append(S)
                # CV dist
                U_test = np.ma.dot(test_obs[i_obs], V).data
                U_test /= np.diag(S)
                reconstruct_test = U_test.dot(S).dot(V.T)
                reconstruct_test = np.ma.array(reconstruct_test,
                                               mask=np.isnan(reconstruct_test))
                obs_error = test_obs[i_obs] - reconstruct_test
                # update the new distortion
                obs_error_data = obs_error.data
                obs_error_data[np.isnan(obs_error_data)] = 0
                error_ = norm(obs_error, 'fro')
                error_ = error_ / np.sqrt(np.sum(~test_obs[i_obs].mask))
                dist_sum_iter.append(error_)
            # CV error
            dists[0][i - 1] = np.mean(dist_sum_iter)
            dists[1][i - 1] = np.std(dist_sum_iter)
            # mean of U and S, ensure same rotation
            X_U = np.mean([u_i.dot(u_i.T)
                           for u_i in sample_loadings], axis=0)
            _, S_shared, _ = svds(X_U, k=self.n_components, which='LM')
            S_shared = np.diag(S_shared)
            S_shared = S_shared / np.linalg.norm(S_shared)
            U_shared = np.average(sample_loadings, axis=0)
            U_shared -= U_shared.mean(0)
            feature_loadings = [(S_shared).dot(v_i.T).T
                                for v_i in feature_loadings]

        # compensates the smaller average size of
        # observed values vs. missing
        S_shared = S_shared / rescal_param
        # ensure the loadings are sorted properly
        idx = np.argsort(np.diag(S_shared))[::-1]
        S_shared = S_shared[idx, :][:, idx]
        U_shared = U_shared[:, idx]
        feature_loadings = [V[:, idx] for V in feature_loadings]

        return U_shared, S_shared, feature_loadings, dists


def svd_sort(U, S, V):
    """
    Sorting via the s matrix from SVD. In addition to
    sign correction from the U matrix to ensure a
    deterministic output.

    Parameters
    ----------
    U: array-like
        U matrix from SVD
    V: array-like
        V matrix from SVD
    S: array-like
        S matrix from SVD

    Notes
    -----
    S matrix can be off diagonal elements.
    """
    # See https://github.com/scikit-learn/scikit-learn/
    # blob/7b136e92acf49d46251479b75c88cba632de1937/sklearn/
    # decomposition/pca.py#L510-#L518 for context.
    # Because svds do not abide by the normal
    # conventions in scipy.linalg.svd/randomized_svd
    # the output has to be reversed
    idx = np.argsort(np.diag(S))[::-1]
    # sorting following the solution
    # provided by https://stackoverflow.com/
    # questions/36381356/sort-matrix-based
    # -on-its-diagonal-entries
    S = S[idx, :][:, idx]
    U = U[:, idx]
    V = V[:, idx]
    return U, S, V


def cost_function(U, V, S, obs, mask, step_size, rho):
    """
    Parameters
    ----------
    U, V, S, obs, mask, step_size, rho

    Notes
    -----
    M ~ imputed
    """
    # shape of subject loading
    n, n_components = U.shape
    # calculate the Frobenius norm between observed values and imputed values
    distortion = np.sum(
        np.sum((np.multiply((U.dot(S).dot(V.T) - obs), mask)**2))) / 2
    # calculate the  cartesian product of two Grassmann manifolds
    V_manifold = rho * grassmann_manifold_one(V, step_size, n_components)
    U_manifold = rho * grassmann_manifold_one(U, step_size, n_components)
    return distortion + V_manifold + U_manifold


def gradient_decent(U, V, S, obs, mask, step_size, rho):
    """
    A single iteration of the gradient decent update to the U and V matrix.

    Parameters
    ----------
    U, V, S, obs, mask, step_size, rho

    Returns
    -------
    U_update, V_update
    """
    # shape of loadings
    n, n_components = U.shape
    m, n_components = V.shape
    # generate the new values imputed
    # from the loadings from obs values
    US = U.dot(S)
    VS = V.dot(S.T)
    imputed = US.dot(V.T)
    # calculate the U and V distortion
    Qu = U.T.dot(np.multiply((obs - imputed), mask)).dot(VS) / n
    Qv = V.T.dot(np.multiply((obs - imputed), mask).T).dot(US) / m
    # create new loadings based on the distortion between obs and imputed
    # pass these loadings to back to the decent.
    U_update = np.multiply((imputed - obs), mask).dot(VS) + U.dot(Qu) + \
        rho * grassmann_manifold_two(U, step_size, n_components)
    V_update = np.multiply((imputed - obs), mask).T.dot(US) + V.dot(Qv) + \
        rho * grassmann_manifold_two(V, step_size, n_components)
    return U_update, V_update


def line_search(
        U,
        U_update,
        V,
        V_update,
        S,
        obs,
        mask,
        step_size,
        rho,
        resolution_limit=20,
        line=-1e-1):
    """
    An exact line search
        gradient decent converging for a quadratic function

    Parameters
    ----------
    U, U_update, V, V_update, S, obs, mask, step_size, rho
    """

    norm_update = norm(U_update, 'fro')**2 + norm(V_update, 'fro')**2
    # this is the resolution limit (line > 2**-20)
    cost = np.zeros(resolution_limit + 1)
    cost[0] = cost_function(U, V, S, obs, mask, step_size, rho)
    for i in range(resolution_limit):
        cost[i +
             1] = cost_function(U +
                                line *
                                U_update, V +
                                line *
                                V_update, S, obs, mask, step_size, rho)
        if ((cost[i + 1] - cost[0]) <= .5 * line * norm_update):
            return line
        line = line / 2
    return line


def singular_values(U, V, obs, mask):
    """
    Generates the singular values from the updated
    U & V loadings for one iteration.

    Parameters
    ----------
    U, V, obs, mask

    """

    n, n_components = U.shape
    C = np.ravel(U.T.dot(obs).dot(V))
    A = np.zeros((n_components * n_components, n_components * n_components))
    for i in range(n_components):
        for j in range(n_components):
            ind = j * n_components + i
            x = U[:, i].reshape(1, len(U[:, i]))
            manifold = V[:, j].reshape(len(V[:, j]), 1)
            tmp = np.multiply((manifold.dot(x)).T, mask)
            temp = U.T.dot(tmp).dot(V)
            A[:, ind] = np.ravel(temp)
    S = np.linalg.lstsq(A, C, rcond=1e-12)[0]
    S = S.reshape((n_components, n_components)).T

    return S


def grassmann_manifold_one(U, step_size, n_components):
    """
    The first Grassmann Manifold

    Parameters
    ----------
    U, step_size, n_components
    """
    # get the step from the manifold
    step = np.sum(U**2, axis=1) / (2 * step_size * n_components)
    manifold = np.exp((step - 1)**2) - 1
    manifold[step < 1] = 0
    manifold[manifold == np.inf] = 0
    return manifold.sum()


def grassmann_manifold_two(U, step_size, n_components):
    """
    The second Grassmann Manifold

    Parameters
    ----------
    U, step_size, n_components


    """
    # get the step from the manifold
    step = np.sum(U**2, axis=1) / (2 * step_size * n_components)
    step = 2 * np.multiply(np.exp((step - 1)**2), (step - 1))
    step[step < 0] = 0
    step = step.reshape(len(step), 1)
    step = np.multiply(U, repmat(step, 1, n_components)) / \
        (step_size * n_components)
    return step
