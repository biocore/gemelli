# ----------------------------------------------------------------------------
# Copyright (c) 2019--, gemelli development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np
from numpy.random import randn
from numpy.linalg import norm
from .base import _BaseImpute
from scipy.spatial import distance


class TenAls(_BaseImpute):

    def __init__(self, rank=3, iteration=50, ninit=50, tol=1e-8):
        """

        This class performs a low-rank 3rd order
        tensor factorization for partially observered
        non-symmetric sets. This method relies on a
        CANDECOMP/PARAFAC (CP) tensor decomposition.
        Missing values are handled by an alternating
        least squares (ALS) minimization between
        TE and TE_hat.

        Parameters
        ----------
        Tensor : array-like
            A 3rd order tensor, often
            compositionally transformed,
            with missing values. The missing
            values must be zeros. Tensor must
            be of the shape:
            first dimension = samples
            second dimension = features
            third dimension = conditions
        r : int, optional
            The underlying low-rank, will be
            equal to the number of rank 1
            components that are output. The
            higher the rank given, the more
            expensive the computation will
            be.
        ninit : int, optional
            The number of initialization
            vectors. Larger values will
            give more accurate factorization
            but will be more computationally
            expensive.
        iteration : int, optional
            Max number of iterations.
        tol : float, optional
            The stopping point in the minimization
            of TE and the factorization between
            each iteration.

        Attributes
        -------
        eigenvalues : array-like
            The singular value vectors (1,r)
        explained_variance_ratio : array-like
            The percent explained by each
            rank-1 factor. (1,r)
        sample_distance : array-like
            The euclidean distance between
            the sample_loading and it'self
            transposed of shape (samples, samples)
        conditional_loading  : array-like
            The conditional loading vectors
            of shape (conditions, r)
        feature_loading : array-like
            The feature loading vectors
            of shape (features, r)
        sample_loading : array-like
            The sample loading vectors
            of shape (samples, r)
        s : array-like
            The r-dimension vector.
        dist : array-like
            A absolute distance vector
            between TE and TE_hat.

        References
        ----------
        .. [1] A. Anandkumar, R. Ge, D. Hsu,
               S. M. Kakade, M. Telgarsky,
               Tensor Decompositions for Learning
               Latent Variable Models
               (A Survey for ALT).
               Lecture Notes in
               Computer Science
               (2015), pp. 19–38.
        .. [2] P. Jain, S. Oh, in Advances in Neural
               Information Processing Systems
               27, Z. Ghahramani, M. Welling,
               C. Cortes, N. D. Lawrence,
               K. Q. Weinberger, Eds.
               (Curran Associates, Inc., 2014),
               pp. 1431–1439.

        Examples
        --------
        >>> r = 3 # rank is 2
        >>> n1 = 10
        >>> n2 = 10
        >>> n3 = 10
        >>> U01 = np.random.rand(n1,r)
        >>> U02 = np.random.rand(n2,r)
        >>> U03 = np.random.rand(n3,r)
        >>> U1, temp = qr(U01)
        >>> U2, temp = qr(U02)
        >>> U3, temp = qr(U03)
        >>> U1=U1[:,0:r]
        >>> U2=U2[:,0:r]
        >>> U3=U3[:,0:r]
        >>> T = np.zeros((n1,n2,n3))
        >>> for i in range(n3):
        >>>     T[:,:,i] = np.matmul(U1,np.matmul(np.diag(U3[i,:]),U2.T))
        >>> p = 2*(r**0.5*np.log(n1*n2*n3))/np.sqrt(n1*n2*n3)
        >>> E = abs(np.ceil(np.random.rand(n1,n2,n3)-1+p))
        >>> E = T*E
        >>> TF = TenAls()
        >>> TF.fit(TE_noise)
        """

        self.rank = rank
        self.iteration = iteration
        self.ninit = ninit
        self.tol = tol
        self.sparse_tensor = None

    def fit(self, Tensor):
        """

        Run _fit() a wrapper
        for the tenals helper.

        Parameters
        ----------
        Tensor : array-like
            A 3rd order tensor, often
            compositionally transformed,
            with missing values. The missing
            values must be zeros. Tensor must
            be of the shape:
            first dimension = samples
            second dimension = features
            third dimension = conditions
        """

        self.sparse_tensor = Tensor.copy()
        self._fit()
        return self

    def _fit(self):
        """
        This function runs the
        tenals helper.

        """

        # make copy for imputation, check type
        sparse_tensor = self.sparse_tensor

        if not isinstance(sparse_tensor, np.ndarray):
            sparse_tensor = np.array(sparse_tensor)
            if not isinstance(sparse_tensor, np.ndarray):
                raise ValueError('Input data is should be type numpy.ndarray')
            if len(sparse_tensor.shape) < 3 or len(sparse_tensor.shape) > 3:
                raise ValueError('Input data is should be 3rd-order tensor',
                                 ' with shape (samples, features, time)')

        if (np.count_nonzero(sparse_tensor) == 0 and
                np.count_nonzero(~np.isnan(sparse_tensor)) == 0):
            raise ValueError('No missing data in the format np.nan or 0')

        if np.count_nonzero(np.isinf(sparse_tensor)) != 0:
            raise ValueError('Contains either np.inf or -np.inf')

        if self.rank > np.max(sparse_tensor.shape):
            raise ValueError('rank must be less than the maximum shape')

        # return tensor decomp
        E = np.zeros(sparse_tensor.shape)
        E[abs(sparse_tensor) > 0] = 1
        loadings, s_, dist = tenals(sparse_tensor, E, r=self.rank,
                                         ninit=self.ninit,
                                         nitr=self.iteration,
                                         tol=self.tol)

        U, V, UV_cond = loadings
        self.eigenvalues = np.diag(s_)
        self.explained_variance_ratio = list(self.eigenvalues / self.eigenvalues.sum())
        self.sample_distance = distance.cdist(U, U)
        self.conditional_loading = UV_cond
        self.feature_loading = V
        self.sample_loading = U
        self.s = s_
        self.dist = dist


def tenals(TE, E, r=3, ninit=50, nitr=50, tol=1e-8):
    """
    A low-rank 3rd order tensor factorization
    for partially observered non-symmetric
    sets. This method relies on a CANDECOMP/
    PARAFAC (CP) tensor decomposition. Missing
    values are handled by  and alternating
    least squares (ALS) minimization between
    TE and TE_hat.

    Parameters
    ----------
    TE : array-like
        A sparse 3rd order tensor with zeros
        in place of missing values. Tensor is
        given in the shape (n1, n2, n3). Where
        n1, n2, and n3 may or may not be equal.
    E : array-like
        A masking array of missing values.
    r : int, optional
        The underlying low-rank, will be
        equal to the number of rank 1
        components that are output. The
        higher the rank given, the more
        expensive the computation will
        be.
    ninit : int, optional
        The number of initialization
        vectors. Larger values will
        give more accurate factorization
        but will be more computationally
        expensive.
    nitr : int, optional
        Max number of iterations.
    tol : float, optional
        The stopping point in the minimization
        of TE and the factorization between
        each iteration.

    Returns
    -------
    V1 : array-like
        The factorization of shape
        (n1, r).
    V2 : array-like
        The factorization of shape
        (n2, r).
    V3 : array-like
        The factorization of shape
        (n3, r).
    S : array-like
        The r-dimension vector.
    dist : array-like
        A absolute distance vector
        between TE and TE_hat.

    Raises
    ------
    ValueError
        Nan values in input, factorization
        did not converge.

    References
    ----------
    .. [1] P. Jain, S. Oh, in Advances in Neural
            Information Processing Systems
            27, Z. Ghahramani, M. Welling,
            C. Cortes, N. D. Lawrence,
            K. Q. Weinberger, Eds.
            (Curran Associates, Inc., 2014),
            pp. 1431–1439.

    Examples
    --------
    >>> r = 3 # rank is 2
    >>> n1 = 10
    >>> n2 = 10
    >>> n3 = 10
    >>> U01 = np.random.rand(n1,r)
    >>> U02 = np.random.rand(n2,r)
    >>> U03 = np.random.rand(n3,r)
    >>> U1, temp = qr(U01)
    >>> U2, temp = qr(U02)
    >>> U3, temp = qr(U03)
    >>> U1=U1[:,0:r]
    >>> U2=U2[:,0:r]
    >>> U3=U3[:,0:r]
    >>> T = np.zeros((n1,n2,n3))
    >>> for i in range(n3):
    >>>     T[:,:,i] = np.matmul(U1,np.matmul(np.diag(U3[i,:]),U2.T))
    >>> p = 2*(r**0.5*np.log(n1*n2*n3))/np.sqrt(n1*n2*n3)
    >>> E = abs(np.ceil(np.random.rand(n1,n2,n3)-1+p))
    >>> E = T*E
    >>> L1,L2,L3,s,dist = tenals(TE_noise,E)

    """

    # start
    n1, n2, n3 = TE.shape
    dims = TE.shape

    # normTE = 0
    # for i3 in range(n3):
    #     normTE = normTE + norm(TE[:, :, i3])**2
    normTE = norm(TE)**2

    # initialization by Robust Tensor Power Method (modified for non-symmetric
    # tensors)
    # U01 = np.zeros((n1, r))
    # U02 = np.zeros((n2, r))
    # U03 = np.zeros((n3, r))
    U = [np.zeros((n, r)) for n in dims]
    U01, U02, U03 = U
    S0 = np.zeros((r, 1))
    for i in range(r):
        tU = [np.zeros((n, ninit)) for n in dims]
        # tU1 = np.zeros((n1, ninit))
        # tU2 = np.zeros((n2, ninit))
        # tU3 = np.zeros((n3, ninit))

        tU1, tU2, tU3 = tU
        tS = np.zeros((ninit, 1))
        for init in range(ninit):
            # [tU1[:, init], tU2[:, init], tU3[:, init]] = RTPM(
            #     TE - CPcomp(S0, U1, U2, U3), max_iter=nitr)
            initializations = RTPM(
                TE - CPcomp(S0, U), max_iter=nitr)

            for tUn_idx, tUn in enumerate(tU):
                tUn[:, init] = initializations[tUn_idx]
                tUn[:, init] = tUn[:, init] / norm(tUn[:, init])
                assert tUn is tU[tUn_idx]
            tU1, tU2, tU3 = tU

            # tU1[:, init] = tU1[:, init] / norm(tU1[:, init])
            # tU2[:, init] = tU2[:, init] / norm(tU2[:, init])
            # tU3[:, init] = tU3[:, init] / norm(tU3[:, init])
            # tS[init] = TenProj(TE - CPcomp(S0, U01, U02, U03),
            #                   tU1[:, [init]], tU2[:, [init]], tU3[:, [init]])
            tS[init] = TenProjAlt(TE - CPcomp(S0, U),
                                  [tUn[:, [init]] for tUn in tU])
            # assert np.allclose(tS[init], tS_oneoff_alt)
            # print(tS[init])

        idx = np.argmax(tS, axis=0)[0]
        for tUn, Un in zip(tU, U):
            Un[:, i] = tUn[:, idx] / norm(tUn[:, idx])
        U01, U02, U03 = U

        # U01[:, i] = tU1[:, idx] / norm(tU1[:, idx])
        # U02[:, i] = tU2[:, idx] / norm(tU2[:, idx])
        # U03[:, i] = tU3[:, idx] / norm(tU3[:, idx])
        # S0[i] = TenProj(TE - CPcomp(S0, U01, U02, U03),
        #                 U01[:, [i]], U02[:, [i]], U03[:, [i]])
        S0[i] = TenProjAlt(TE - CPcomp(S0, U),
                           [Un[:, [i]] for Un in U])
        # print(S0[i])
        # print(TenProj(TE - CPcomp(S0, *U),
        #                *[Un[:, [i]] for Un in U]))

    # apply alternating least squares
    # V1 = U01.copy()
    # V2 = U02.copy()
    # V3 = U03.copy()
    V = [Un.copy() for Un in U]
    V_alt = [Un.copy() for Un in U]
    V1, V2, V3 = V
    S = S0.copy()
    # corresponds to line 5 of pseudo code
    for itrs in range(nitr):
        # corresponds to line 7 of pseudo code
        for q in range(r):
            S_ = S.copy()
            S_[q] = 0
            S_alt = S.copy()
            S_alt[q] = 0
            A = np.multiply(CPcomp(S_, V), E)
            # v1 = V1[:, q].copy()
            # v2 = V2[:, q].copy()
            # v3 = V3[:, q].copy()
            v = [Vn[:, q].copy() for Vn in V]
            v1, v2, v3 = v
            v_alt = [Vn[:,q].copy() for Vn in V_alt]
            for Vn in V:
                Vn[:, q] = 0
            # V1[:, q] = 0
            # V2[:, q] = 0
            # V3[:, q] = 0

            # n1 and n2 are components of shape
            den1 = np.zeros((n1, 1))
            den2 = np.zeros((n2, 1))

            # den should en up as a list of np.zeros((dim_i, 1))
            den = [np.zeros(dim) for dim in dims]

            for dim, dim_size in enumerate(dims):
                dims_np = np.arange(len(dims))
                dot_across = dims_np[dims_np != dim]
                v_dim = np.tensordot(TE - A,
                                     v_alt[dot_across[0]],
                                     axes=(1 if dim == 0 else 0, 0))
                den[dim] = np.tensordot(E,
                                        v_alt[dot_across[0]]**2,
                                        axes=(1 if dim == 0 else 0, 0))

                for inner_dim in dot_across[1:]:
                    v_dim = np.tensordot(v_dim,
                                         v_alt[inner_dim],
                                         axes=(1 if inner_dim > dim else 0, 0))
                    den[dim] = np.tensordot(den[dim],
                                            v_alt[inner_dim]**2,
                                            axes=(1 if inner_dim > dim else
                                                  0, 0))

                #den[dim]#.reshape(dims[dim], 1)

                v_alt[dim] = V[dim][:, q] + v_dim.flatten()
                v_alt[dim] = v_alt[dim] / den[dim]

                if dim == len(dims) - 1:
                    S_alt[q] = norm(v_alt[dim])

                v_alt[dim] = v_alt[dim] / norm(v_alt[dim])
                V_alt[dim][:, q] = v_alt[dim]

            for i3 in range(n3):
                # REMINDER np.multiply is element-wise
                # `A` is CPD reconstruction of TE
                # TODO use tensordot like in RTPM
                V1[:, q] = V1[:, q] + \
                    np.multiply(v3[i3],
                                np.matmul((TE[:, :, i3]
                                           - A[:, :, i3]),
                                          v2)).flatten()
                den1 = den1 + \
                    np.multiply(v3[i3]**2, np.matmul(E[:, :, i3],
                                                     v2**2)).reshape(
                                            dims[0], 1)
            # TODO code kind of works to here
            assert np.allclose(den[0], den1.flatten())
            # assert np.allclose(v_alt[0], V1[:, q])
            # assert den[dim].shape == (n1, 1)

            v1 = V1[:, q].reshape(dims[0], 1) / den1
            v1 = v1 / norm(v1)

            assert np.allclose(v_alt[0], v1.flatten())

            # TODO use tensordot like in RTPM
            for i3 in range(n3):
                V2[:, q] = V2[:, q] + \
                    np.multiply(v3[i3],
                                np.matmul((TE[:, :, i3]
                                           - A[:, :, i3]).T,
                                          v1)).flatten()
                den2 = den2 + \
                    np.multiply(v3[i3]**2, np.matmul(E[:, :, i3].T,
                                                     v1**2)).reshape(
                                            dims[1], 1)
            v2 = V2[:, q].reshape(dims[1], 1) / den2
            v2 = v2 / norm(v2)

            assert np.allclose(den[1], den2.flatten())
            assert np.allclose(v_alt[1], v2.flatten())

            # TODO use tensordot like in RTPM
            for i3 in range(n3):
                V3[i3, q] = (np.matmul(v1.T,
                                       np.matmul(TE[:, :, i3]
                                                 - A[:, :, i3], v2)) /
                             np.matmul(np.matmul((v1**2).T,
                                                 (E[:, :, i3])),
                                       (v2**2))).flatten()

            V1[:, q] = v1.flatten()
            V2[:, q] = v2.flatten()
            S[q] = norm(V3[:, q])
            V3[:, q] = V3[:, q] / norm(V3[:, q])

            #print(V_alt[2][:,q], V3[:,q])
            assert np.allclose(v_alt[2], V3[:, q])
            #print(V_alt[2][:,q], V3[:,q])
            #print(S_alt[q], S[q])
            assert np.allclose(S_alt[q], S[q])


            V = V1, V2, V3

        ERR = TE - E * CPcomp(S, V)

        # normERR = 0
        # for i3 in range(n3):
        #    normERR = normERR + norm(ERR[:, :, i3])**2

        normERR = norm(ERR)**2
        if np.sqrt(normERR / normTE) < tol:
            break
    dist = np.sqrt(normERR / normTE)
    V = V1, V2, V3
    # check that the factorization converged
    if any(sum(sum(np.isnan(Vn))) > 0 for Vn in V):
        raise ValueError("The factorization did not converge.",
                         "Please check the input tensor for errors.")

    S = np.diag(S.flatten())
    # sort the eigenvalues
    idx = np.argsort(np.diag(S))[::-1]
    S = S[idx, :][:, idx]
    # sort loadings
    loadings = [Vn[:, idx] for Vn in V]
    # V1 = V1[:, idx]
    # V2 = V2[:, idx]
    # V3 = V3[:, idx]
    # loadings = [V1, V2, V3]

    return loadings, S, dist


def RTPM(tensor, max_iter=50):
    """
    TODO finish generalization

    The Robust Tensor Power Method
    (RTPM). Is a generalization of
    the widely used power method for
    computing lead singular values
    of a matrix and can approximate
    the largest singular vectors of
    a tensor.

    Parameters
    ----------
    TODO generalize tensor
    tensor : array-like
        tensor of shape
        (n1, n2, n3).
    max_iter : int
        maximum iterations.

    Returns
    -------
    TODO generalize output
    u1 : array-like
        The singular vectors n1
    u2 : array-like
        The singular vectors n2
    u3 : array-like
        The singular vectors n3

    References
    ----------
    .. [1] A. Anandkumar, R. Ge, D. Hsu,
            S. M. Kakade, M. Telgarsky,
            Tensor Decompositions for Learning
            Latent Variable Models
            (A Survey for ALT).
            Lecture Notes in
            Computer Science
            (2015), pp. 19–38.
    .. [2] P. Jain, S. Oh, in Advances in Neural
            Information Processing Systems
            27, Z. Ghahramani, M. Welling,
            C. Cortes, N. D. Lawrence,
            K. Q. Weinberger, Eds.
            (Curran Associates, Inc., 2014),
            pp. 1431–1439.
    TODO cite Guaranteed Non-Orthogonal Tensor Decomposition via
     Alternating Rank-1 Updates

    """

    # RTPM
    n_dims = len(tensor.shape)

    all_u = [randn(n, 1) for n in tensor.shape]
    all_u = [vec / norm(vec) for vec in all_u]
    for itr in range(max_iter):
        # tensordot generalization to higher dims
        v = []
        dims = np.arange(n_dims)
        for dim in dims:
            dot_across = dims[dims != dim]
            v_dim = np.tensordot(tensor,
                                 all_u[dot_across[0]],
                                 axes=(1 if dim == 0 else 0, 0))
            for inner_dim in dot_across[1:]:
                v_dim = np.tensordot(v_dim,
                                     all_u[inner_dim],
                                     axes=(1 if inner_dim > dim else 0, 0))
            v.append(v_dim)

        v = [v_n.reshape(v_n.shape[:-1]) for v_n in v]

        all_u_previous = [u for u in all_u]
        all_u = [v_i / norm(v_i) for v_i in v]

        if sum(norm(u0 - u) for u0, u in zip(all_u_previous, all_u)) < 1e-7:
            break

    return tuple(u.flatten() for u in all_u)


def CPcomp(S, U):
    """
    TODO variable names
    This function takes the
    CP decomposition of a 3rd
    order tensor and outputs
    the reconstructed tensor
    TE_hat.

    Parameters
    ----------
    U : list of array-like
        Element i is a factor of shape
        (n[i], r).
    S : array-like
        The r-dimension vector.

    Returns
    -------
    T : array-like
        TE_hat of shape
        tuple(n[i] for i in range(len(U))).
    """

    output_shape = tuple(u.shape[0] for u in U)
    to_multiply = [S.T*u if i == 0 else u for i, u in enumerate(U)]
    product = khatri_rao(to_multiply)
    T = product.sum(1).reshape(output_shape)

    return T


def TenProjAlt(D, U_list):
    """
    TODO make documentation better
    The Orthogonal tensor
    projection created by
    the TE - TE_hat distance.
    Used in the initialization
    step with RTPM.

    Parameters
    ----------
    D : array-like
        shape (n1,n2,n3)
    U_list : list of array-like
        Element i is a factor of shape
        (n[i], r).

    Returns
    -------
    M : float
        TODO ???.
    """
    current = D
    for u in U_list:
        current = np.tensordot(current, u, axes=(0, 0))
    return current


def khatri_rao(matrices):
    # TODO document and cite
    # FROM TENSORLY
    n_columns = matrices[0].shape[1]
    n_factors = len(matrices)

    start = ord('a')
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(n_factors))
    source = ','.join(i + common_dim for i in target)
    operation = source + '->' + target + common_dim

    return np.einsum(operation, *matrices).reshape((-1, n_columns))
