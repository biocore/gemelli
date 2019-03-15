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
#from .tenals import  tenals
from scipy.spatial import distance
import warnings

class TenAls(_BaseImpute):

    def __init__(self, rank=3, iteration=50, ninit=50, tol=1e-8):
        """

        Description.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        ValueError

        Warning

        References
        ----------

        Examples
        --------

        """

        self.rank = rank
        self.iteration = iteration
        self.ninit = ninit
        self.tol = tol

        return

    def fit(self, X):

        """

        Description.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        ValueError

        Warning

        References
        ----------

        Examples
        --------

        """

        X_sparse = X.copy().astype(np.float64)
        self.X_sparse = X_sparse
        self._fit()
        return self

    def _fit(self):

        """

        Description.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        ValueError

        Warning

        References
        ----------

        Examples
        --------

        """

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
        self.dist = dist

    def fit_transform(self, X):

        """

        Description.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        ValueError

        Warning

        References
        ----------

        Examples
        --------

        """
    
        X_sparse = X.copy().astype(np.float64)
        self.X_sparse = X_sparse
        self._fit()
        return self.sample_loading, self.feature_loading, self.time_loading, self.s, self.dist

def tenals(TE, E, r = 3, ninit = 50, nitr = 50, tol = 1e-8):

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
    .. [1] P. Jain, S. Oh, in Advances in Neural Information Processing
           Systems 27, Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence,
           K. Q. Weinberger, Eds. (Curran Associates, Inc., 2014), pp. 1431–1439.

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
  
    #start
    n1,n2,n3 = TE.shape
    p = np.count_nonzero(TE)

    normTE = 0 
    for i3 in range(n3):
        normTE = normTE + norm(TE[:,:,i3])**2

    # initialization by Robust Tensor Power Method (modified for non-symmetric tensors)
    U01 = np.zeros((n1,r))
    U02 = np.zeros((n2,r))
    U03 = np.zeros((n3,r))
    S0 = np.zeros((r,1))
    for i in range(r):
        tU1 = np.zeros((n1,ninit))
        tU2 = np.zeros((n2,ninit))
        tU3 = np.zeros((n3,ninit))
        tS = np.zeros((ninit,1))
        for init in range(ninit):
            [tU1[:,init], tU2[:,init], tU3[:,init]] = RTPM(TE-CPcomp(S0,U01,U02,U03), max_iter=nitr)  
            tU1[:,init] = tU1[:,init]/norm(tU1[:,init])
            tU2[:,init] = tU2[:,init]/norm(tU2[:,init])
            tU3[:,init] = tU3[:,init]/norm(tU3[:,init])
            tS[init] = TenProj(TE-CPcomp(S0,U01,U02,U03),tU1[:,[init]],tU2[:,[init]],tU3[:,[init]])
        [C, I] = np.max(tS,axis=0)[0], np.argmax(tS, axis=0)[0]
        U01[:,i] = tU1[:,I]/norm(tU1[:,I])
        U02[:,i] = tU2[:,I]/norm(tU2[:,I])
        U03[:,i] = tU3[:,I]/norm(tU3[:,I])
        S0[i] = TenProj(TE-CPcomp(S0,U01,U02,U03),U01[:,[i]],U02[:,[i]],U03[:,[i]])

    # apply alternating least squares        
    V1 = U01.copy()
    V2 = U02.copy()
    V3 = U03.copy()
    S = S0.copy()
    for itrs in range(nitr):
        for q in range(r):
            S_ = S.copy()
            S_[q] = 0 
            A = np.multiply(CPcomp(S_,V1,V2,V3),E)
            v1 = V1[:,q].copy()
            v2 = V2[:,q].copy()
            v3 = V3[:,q].copy()
            V1[:,q] = 0
            V2[:,q] = 0
            V3[:,q] = 0
            den1 = np.zeros((n1,1))
            den2 = np.zeros((n2,1))
            s = S[q]
            for i3 in range(n3):
                V1[:,q] = V1[:,q] + np.multiply(v3[i3],np.matmul((TE[:,:,i3]-A[:,:,i3]),v2))
                den1 = den1 + np.multiply(v3[i3]**2,np.matmul(E[:,:,i3],v2*v2)).reshape(den1.shape[0],1)        
            v1 = V1[:,q].reshape(den1.shape[0],1)/den1
            v1 = v1/norm(v1)
            for i3 in range(n3):
                V2[:,q] = V2[:,q] + np.multiply(v3[i3],np.matmul((TE[:,:,i3]-A[:,:,i3]).T,v1)).flatten()
                den2 = den2 + np.multiply(v3[i3]**2,np.matmul(E[:,:,i3].T,np.multiply(v1,v1)))
            v2 = V2[:,q].reshape(den2.shape[0],1)/den2
            v2 = v2/norm(v2) 
            for i3 in range(n3):
                V3[i3,q] = (np.matmul(v1.T,np.matmul(TE[:,:,i3]-A[:,:,i3],v2))/np.matmul(np.matmul((v1*v1).T,(E[:,:,i3])),(v2*v2))).flatten()        
            V1[:,q] = v1.flatten()
            V2[:,q] = v2.flatten()
            S[q] = norm(V3[:,q])
            V3[:,q] = V3[:,q]/norm(V3[:,q]) 
        ERR = TE - E*CPcomp(S,V1,V2,V3)
        normERR = 0  
        for i3 in range(n3):
            normERR = normERR + norm(ERR[:,:,i3])**2
        if np.sqrt(normERR/normTE) < tol: 
            break
    dist = np.sqrt(normERR/normTE)
    # check that the fact. converged
    if sum(sum(np.isnan(V1))) > 0 or\
       sum(sum(np.isnan(V2))) > 0 or\
       sum(sum(np.isnan(V3))) > 0:
        raise ValueError("The factorization did not converge.",
                         "Please check the input tensor for errors.")

    return V1[:,::-1], V2[:,::-1], V3[:,::-1], np.diag(S.flatten()), dist

def RTPM(T, max_iter = 50):

    """

    The Robust Tensor Power Method
    (RTPM). Is a generalization of
    the widely used power method for
    computing lead singular values
    of a matrix and can approximate
    the largest singular vectors of
    a tensor.

    Parameters
    ----------
    T : array-like
        tensor of shape
        (n1, n2, n3).
    max_iter : int
        maximum iterations.

    Returns
    -------
    u1 : array-like
        The singular vectors n1
    u2 : array-like
        The singular vectors n2
    u3 : array-like
        The singular vectors n3

    References
    ----------
    .. [1] A. Anandkumar, R. Ge, D. Hsu, S. M. Kakade, M. Telgarsky, 
           Tensor Decompositions for Learning Latent Variable Models
           (A Survey for ALT). Lecture Notes in Computer Science
           (2015), pp. 19–38.
    .. [2] P. Jain, S. Oh, in Advances in Neural Information Processing
           Systems 27, Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence,
           K. Q. Weinberger, Eds. (Curran Associates, Inc., 2014), pp. 1431–1439.
    """

    #RTPM
    n1, n2, n3 = T.shape
    u1 = randn(n1,1)/norm(randn(n1,1))
    u2 = randn(n2,1)/norm(randn(n2,1))
    u3 = randn(n3,1)/norm(randn(n3,1))
    #conv
    for itr in range(max_iter):
        v1 = np.zeros((n1,1))
        v2 = np.zeros((n2,1))
        v3 = np.zeros((n3,1))
        for i3 in range(n3):
            v3[i3] = np.matmul(np.matmul(u1.T,T[:,:,i3]),u2)
            v1 = v1 + np.matmul(u3[i3][0]*T[:,:,i3],u2)
            v2 = v2 + np.matmul(u3[i3][0]*T[:,:,i3].T,u1)
        u10 = u1
        u1 = v1/norm(v1)
        u20 = u2
        u2 = v2/norm(v2)
        u30 = u3
        u3 = v3/norm(v3)
        if(norm(u10-u1)+norm(u20-u2)+norm(u30-u3)<1e-7) :
            break

    return u1.flatten(),u2.flatten(),u3.flatten()

def CPcomp(S,U1,U2,U3):

    """
    This function takes the
    CP decomposition of a 3rd
    order tensor and outputs
    the reconstructed tensor
    TE_hat.

    Parameters
    ----------
    U1 : array-like
        The factorization of shape
        (n1, r).
    U2 : array-like
        The factorization of shape
        (n2, r).
    U3 : array-like
        The factorization of shape
        (n3, r).
    S : array-like
        The r-dimension vector.

    Returns
    -------
    T : array-like
        TE_hat of shape
        (n1, n2, n3).
    """
  
    ns, rs = S.shape
    n1, r1 = U1.shape
    n2, r2 = U2.shape
    n3, r3 = U3.shape
    r = min([rs, r1, r2, r3])
    T = np.zeros((n1,n2,n3))
    for i in range(n3):
        t_i = np.diag(np.multiply(U3[i,:],S.T)[0])
        T[:,:,i] = np.matmul(np.matmul(U1,t_i),U2.T)
    return T

def TenProj(D, U1, U2, U3):

    """
    The Orthogonal tensor
    projection created by
    the TE - TE_hat distance.
    Used in the initialization
    step with RTPM.

    Parameters
    ----------
    D : array-like
        shape (n1,n2,3)
    U1 : array-like
        The factorization of shape
        (n1, r).
    U2 : array-like
        The factorization of shape
        (n2, r).
    U3 : array-like
        The factorization of shape
        (n3, r).

    Returns
    -------
    M : array-like
        Projection.
    """

    n1, r1 = U1.shape
    n2, r2 = U2.shape
    n3, r3 = U3.shape
    M = np.zeros((r1,r2,r3))
    for i in range(r3):
        A = np.zeros((n1,n2))
        for j in range(n3):
            A = A + D[:,:,j]*U3[j,i]
        M[:,:,i] = np.matmul(np.matmul(U1.T,A),U2)
    return M
