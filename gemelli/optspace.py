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
        mask = (np.abs(obs) > 0).astype(np.int)
        # save the shape of the matrix
        n, m = obs.shape
        # get a measure of sparsity level
        total_nonzeros = np.count_nonzero(mask)
        eps = total_nonzeros / np.sqrt(m * n)
        if isinstance(self.n_components, str):
            if self.n_components.lower() == 'auto':
                # estimate the rank of the matrix
                self.n_components = rank_estimate(obs, eps)
            else:
                raise ValueError("n-components must be an "
                                 "integer or 'auto'.")
        # raise future warning if hard set
        elif isinstance(self.n_components, int):
            if self.n_components > (min(n, m) - 1):
                raise ValueError("n-components must be at most"
                                 " 1 minus the min. shape of the"
                                 " input matrix.")
        # otherwise rase an error.
        else:
            raise ValueError("n-components must be "
                             "an interger or 'auto'")
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
            if(dist[i + 1] < self.tol):
                break
        # compensates the smaller average size of
        # observed values vs. missing
        S = S / rescal_param
        # ensure the loadings are sorted properly
        U, S, V = svd_sort(U, S, V)
        return U, S, V


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
        if((cost[i + 1] - cost[0]) <= .5 * line * norm_update):
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


def rank_estimate(obs, eps, k=20, lam=0.05,
                  min_rank=2, max_iter=5000):

    """
    This function estimates the rank of a
    sparse matrix (i.e. with missing values).

    Parameters
    ----------
    obs: numpy.ndarray - a rclr preprocessed matrix of shape (M,N)
        with missing values set to zero or np.nan.
        N = Features (i.e. OTUs, metabolites)
        M = Samples

    eps: float - Measure of the level of sparsity
        Equivalent to obs N-non-zeros / sqrt(obs.shape)

    k: int - Max number of singular values / rank

    lam: float - Step in the iteration

    min_rank: int - The min. rank allowed

    Returns
    -------
    The estimated rank of the matrix.

    References
    ----------
    .. [1] Part C in Keshavan, R. H., Montanari,
           A. & Oh, S. Low-rank matrix completion
           with noisy observations: A quantitative
           comparison. in 2009 47th Annual Allerton
           Conference on Communication, Control,
           and Computing (Allerton) 1216–1222 (2009).
    """

    # dim. of the data
    n, m = obs.shape
    # get N-singular values
    s = svds(obs,  min(k, n, m) - 1, which='LM',
             return_singular_vectors=False)[::-1]
    # get N+1 singular values
    s_one = s[:-1] - s[1:]
    # simplify iterations
    s_one_ = s_one / np.mean(s_one[-10:])
    # iteration one
    r_one = 0
    iter_ = 0
    while r_one <= 0:
        cost = []
        # get the cost
        for idx in range(s_one_.shape[0]):
            cost.append(lam * max(s_one_[idx:]) + idx)
        # estimate the rank
        r_one = np.argmin(cost)
        lam += lam
        iter_ += 1
        if iter_ > max_iter:
            break

    # iteration two
    cost = []
    # estimate the rank
    for idx in range(s.shape[0] - 1):
        cost.append((s[idx + 1] + np.sqrt(idx * eps) * s[0] / eps) / s[idx])
    r_two = np.argmin(cost)
    # return the final estimate
    return max(r_one, r_two, min_rank)
