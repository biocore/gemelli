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


class TensorFactorization(_BaseImpute):

    def __init__(
            self,
            n_components=3,
            max_als_iterations=50,
            tol_als=1e-8,
            max_rtpm_iterations=50,
            n_initializations=50,
            tol_rtpm=1e-7,
            fillna=1.0):
        """

        This class performs a low-rank N-order
        tensor factorization for partially observered
        non-symmetric sets [1]. This method relies on a
        CANDECOMP/PARAFAC (CP) tensor decomposition [2,3].
        Missing values are handled by an alternating
        least squares (ALS) minimization between
        tensor and the reconstructed tensor.

        Parameters
        ----------
        n_components : int, optional
            The underlying low-rank, will be
            equal to the number of rank 1
            components that are output. The
            higher the rank given, the more
            expensive the computation will
            be.
        max_als_iterations : int, optional
            Max number of Alternating Least Square (ALS).
        tol_als : float, optional
            The minimization -- convergence break point for
            ALS.
        max_rtpm_iterations : int, optional
            Max number of Robust Tensor Power Method (RTPM)
            iterations.
        n_initializations : int, optional
            The number of initialization
            vectors. Larger values will
            give more accurate factorization
            but will be more computationally
            expensive.
        tol_rtpm : float, optional
            The minimization -- convergence break point for
            RTPM.
        fillna : int, optional
            Prevent division by zero in denominator causing nan
            in the optimization iterations.

        Attributes
        ----------
        eigvals : array-like
            The singular value vectors (1,n_components)
        proportion_explained : array-like
            The percent explained by each
            rank-1 factor. (1,n_components)
        loadings : list of array-like
            A list of loadings for all dimensions
            of the data
        subjects : array-like
            The sample loading vectors
            of shape (subjects, n_components)
        features : array-like
            The feature loading vectors
            of shape (features, n_components)
        conditions  : array-like or list of array-like
            The conditional loading vectors
            of shape (conditions, n_components) if there is 1 type
            of condition, and a list of such matrices if
            there are more than 1 type of condition
        subject_trajectory : list of array-like
            The loadings of the dot product between
            subjects and each condition in conditions.
            For each condition in conditions the
            loading is of shape (subjects, condition).
        feature_trajectory : list of array-like
            The loadings of the dot product between
            features and each condition in conditions.
            For each condition in conditions the
            loading is of shape (features, condition).
        subject_distances : list of array-like
            The euclidean distances of the subject trajectory
            for each condition in conditions.
            For each condition in conditions the
            loading is of shape (subjects, subjects).
        feature_distances : list of array-like
            The euclidean distances of the feature trajectory
            for each condition in conditions.
            For each condition in conditions the
            loading is of shape (features, features).
        reconstructed_distance : array-like
            A absolute distance vector
            between tensor and reconstructed tensor.

        References
        ----------
        .. [1] Jain, Prateek, and Sewoong Oh. 2014.
               “Provable Tensor Factorization with Missing Data.”
               In Advances in Neural Information Processing Systems
               27, edited by Z. Ghahramani, M. Welling, C. Cortes,
               N. D. Lawrence, and K. Q. Weinberger, 1431–39.
               Curran Associates, Inc.
        .. [2] A. Anandkumar, R. Ge, M., Janzamin,
               Guaranteed Non-Orthogonal Tensor
               Decomposition via Alternating Rank-1
               Updates. CoRR (2014),
               pp. 1-36.
        .. [3] A. Anandkumar, R. Ge, D. Hsu,
               S. M. Kakade, M. Telgarsky,
               Tensor Decompositions for Learning Latent Variable Models
               (A Survey for ALT). Lecture Notes in Computer Science
               (2015), pp. 19–38.

        Examples
        --------
        TODO
        """

        # save all input perameters as attributes
        self.n_components = n_components
        self.max_als_iterations = max_als_iterations
        self.max_rtpm_iterations = max_rtpm_iterations
        self.n_initializations = n_initializations
        self.tol_als = tol_als
        self.tol_rtpm = tol_rtpm
        self.fillna = fillna

    def fit(self, tensor):
        """

        Run _fit() a wrapper
        for the tenals helper.

        Parameters
        ----------
        tensor : array-like
            A tensor, often
            compositionally transformed,
            with missing values. The missing
            values must be zeros. Canonically,
            Tensor must be of the shape:
            first dimension = subjects
            second dimension = features
            rest dimensions = types of conditions
        """

        self.tensor = tensor.copy()
        self._fit()
        return self

    def _fit(self):
        """
        This function runs the
        tenals helper.

        """

        # make copy for imputation, check type
        tensor = self.tensor
        # ensure that the data is in the format of np.ndarray.
        if not isinstance(tensor, np.ndarray):
            raise ValueError('Input data is should be type numpy.ndarray')
        # ensure the data contains missing values.
        # other methods would be better in the case of fully dense data
        if (np.count_nonzero(tensor) == np.product(tensor.shape) and
                np.count_nonzero(~np.isnan(tensor)) == np.product(tensor.shape)):
            raise ValueError('No missing data in the format np.nan or 0.')
        # ensure there are no undefined values in the array
        if np.count_nonzero(np.isinf(tensor)) != 0:
            raise ValueError('Contains either np.inf or -np.inf')
        # ensure there are less components that the max. shape of the tensor
        if self.n_components > np.max(tensor.shape):
            raise ValueError(
                'n_components must be less than the maximum shape')
        # obtain mask array where values of tensor are zero
        mask = np.zeros(tensor.shape)  # mask of zeros the same shape
        mask[abs(tensor) > 0] = 1  # set masked (missing) values to one
        # run the tensor factorization method [2]
        loadings, eigvals, reconstructed_distance = tenals(tensor,
                                                           mask,
                                                           n_components=self.n_components,
                                                           n_initializations=self.n_initializations,
                                                           max_als_iterations=self.max_als_iterations,
                                                           max_rtpm_iterations=self.max_rtpm_iterations,
                                                           tol_als=self.tol_als,
                                                           tol_rtpm=self.tol_rtpm,
                                                           fillna=self.fillna)
        # save all raw laodings as attribute
        self.loadings = loadings
        # all eigen values
        self.eigvals = np.diag(eigvals)
        # the distance between tensor_imputed and tensor
        self.reconstructed_distance = reconstructed_distance
        # the proortion explained for n_components
        self.proportion_explained = \
            list(self.eigvals / self.eigvals.sum())
        # save array of loadings for subjects
        self.subjects = loadings[0]
        # save array of loadings for features
        self.features = loadings[1]
        # save list of array(s) of loadings for conditions
        self.conditions = loadings[2] if len(loadings[2:]) == 1 \
            else loadings[2:]
        # generate the trajectory(s) and distances
        # list of each condition-subject trajectory
        self.subject_trajectory = []
        # list of each condition-subject distance
        self.subject_distances = []
        # list of each condition-feature trajectory
        self.feature_trajectory = []
        # list of each condition-feature distance
        self.feature_distances = []
        # for each condition in conditions
        # generate a trajectory and distance array
        for condition in loadings[2:]:
            # temporary list of components in subject trajectory
            subject_temp_trajectory = []
            # temporary list of components in feature trajectory
            feature_temp_trajectory = []
            # for each component in the rank given to TensorFactorization
            for component in range(self.n_components):
                # component condition-subject trajectory
                subject_temp_trajectory.append(np.dot(loadings[0][:, [component]],
                                                      condition[:, [component]].T).flatten())
                # component condition-feature trajectory
                feature_temp_trajectory.append(np.dot(loadings[1][:, [component]],
                                                      condition[:, [component]].T).flatten())
            # combine all n_components
            subject_temp_trajectory = np.array(subject_temp_trajectory).T
            feature_temp_trajectory = np.array(feature_temp_trajectory).T
            # save subject-condition trajectory and distance matrix
            self.subject_trajectory.append(subject_temp_trajectory)
            self.subject_distances.append(
                distance.cdist(
                    subject_temp_trajectory,
                    subject_temp_trajectory))
            # save feature-condition trajectory and distance matrix
            self.feature_trajectory.append(feature_temp_trajectory)
            self.feature_distances.append(
                distance.cdist(
                    feature_temp_trajectory,
                    feature_temp_trajectory))


def tenals(
        tensor,
        mask,
        n_components=3,
        n_initializations=50,
        max_als_iterations=50,
        max_rtpm_iterations=50,
        tol_als=1e-8,
        tol_rtpm=1e-8,
        fillna=1.0):
    """
    A low-rank 3rd order tensor factorization
    for partially observered non-symmetric
    sets. This method relies on a CANDECOMP/
    PARAFAC (CP) tensor decomposition. Missing
    values are handled by  and alternating
    least squares (ALS) minimization between
    tensor and reconstructed_tensor.

    Parameters
    ----------
    tensor : array-like
        A sparse `n` order tensor with zeros
        in place of missing values.
    mask : array-like
        A masking array of missing values.
    n_components : int, optional
        The underlying low-rank, will be
        equal to the number of rank 1
        components that are output. The
        higher the rank given, the more
        expensive the computation will
        be.
    max_als_iterations : int, optional
        Max number of Alternating Least Square (ALS).
    tol_als : float, optional
        The minimization -- convergence break point for
        ALS.
    max_rtpm_iterations : int, optional
        Max number of Robust Tensor Power Method (RTPM)
        iterations.
    n_initializations : int, optional
        The number of initialization
        vectors. Larger values will
        give more accurate factorization
        but will be more computationally
        expensive.
    tol_rtpm : float, optional
        The minimization -- convergence break point for
        RTPM.
    fillna : int, optional
        Prevent division by zero in denominator causing nan
        in the optimization iterations.

    Returns
    -------
    loadings : list array-like
        The factors of tensor. The `i`th entry of loadings corresponds to
        the mode-`i` factors of tensor and hase shape (tensor.shape[i], r).
    eigvals : array-like
        The r-dimension vector of eigvals.
    reconstructed_distance : array-like
        A absolute distance vector
        between tensor and reconstructed tensor.

    Raises
    ------
    ValueError
        Nan values in input, factorization
        did not converge.

    References
    ----------
    .. [1] Jain, Prateek, and Sewoong Oh. 2014.
            “Provable Tensor Factorization with Missing Data.”
            In Advances in Neural Information Processing Systems
            27, edited by Z. Ghahramani, M. Welling, C. Cortes,
            N. D. Lawrence, and K. Q. Weinberger, 1431–39.
            Curran Associates, Inc.
    .. [2] A. Anandkumar, R. Ge, M., Janzamin,
            Guaranteed Non-Orthogonal Tensor
            Decomposition via Alternating Rank-1
            Updates. CoRR (2014),
            pp. 1-36.
    .. [3] A. Anandkumar, R. Ge, D. Hsu,
            S. M. Kakade, M. Telgarsky,
            Tensor Decompositions for Learning Latent Variable Models
            (A Survey for ALT). Lecture Notes in Computer Science
            (2015), pp. 19–38.

    Examples
    --------
    TODO
    """

    # Get the shape of tensor to iterate
    tensor_dimensions = tensor.shape
    # Frobenius norm initialization for ALS minimization.
    initial_tensor_frobenius_norm = norm(tensor)**2
    # initialization by Robust Tensor Power Method (modified for non-symmetric
    # tensors).
    initial_eigvals, initial_loadings = robust_tensor_power_method(
        tensor, n_components, n_initializations, max_rtpm_iterations, tol_rtpm)
    # Begin alternating least squares minimization below!
    # make a copy of inital factorization from RTPM
    # to use in the ALS step.
    loadings = initial_loadings.copy()
    eigvals = initial_eigvals.copy()
    # Iterate minimization for at maximum max_als_iterations
    # can break if converges and satisfies tol_als.
    for iteration in range(max_als_iterations):
        # For each rank-1 vector in n_components (total rank)
        for component in range(n_components):
            # set all eigvals to zero for iteration
            eigvals[component] = 0
            # reconstruct the tensor from the loadings to compare to the
            # original tensor
            reconstructed_tensor = np.multiply(
                reconstruct_tensor(eigvals, loadings), mask)
            # generate copy of loadings to reconstruct on each iterations
            loadings_iteration = [loading[:, component].copy()
                                  for loading in loadings]
            # denominator should end up as a list of np.zeros((dim_i, 1))
            denominator = [np.zeros(dim) for dim in tensor_dimensions]
            # for each  dimension of the tensor optimize that dimensions loading
            # based on the distance between the original tensor and the
            # reconstruction
            for dim, dim_size in enumerate(tensor_dimensions):
                # set previous loading to zero
                loadings[dim][:, component] = 0
                # generate indices to perform dot-product across
                dims_np = np.arange(len(tensor_dimensions))
                dot_across = dims_np[dims_np != dim]
                # outer dot-product
                dim_loadings_reconstructed = np.tensordot(tensor - reconstructed_tensor,
                                                          loadings_iteration[dot_across[0]],
                                                          axes=(1 if dim == 0 else 0, 0))
                denominator[dim] = np.tensordot(mask,
                                                loadings_iteration[dot_across[0]]**2,
                                                axes=(1 if dim == 0 else 0, 0))
                # inner dot-product
                for inner_dim in dot_across[1:]:
                    dim_loadings_reconstructed = np.tensordot(
                        dim_loadings_reconstructed, loadings_iteration[inner_dim], axes=(
                            1 if inner_dim > dim else 0, 0))
                    denominator[dim] = np.tensordot(
                        denominator[dim],
                        loadings_iteration[inner_dim]**2,
                        axes=(
                            1 if inner_dim > dim else 0,
                            0))
                # update iteration's loadings
                loadings_iteration[dim] = loadings[dim][:,
                                                        component] + dim_loadings_reconstructed.flatten()
                # Add fillna to prevent division by zero in denominator causing nan.
                # This can occur in early iteration from all zero fibers in the tensor along dim.
                # In practice this should be rare but can occur in very sparse
                # tensors.
                denominator[dim][denominator[dim] == 0] = fillna
                loadings_iteration[dim] = loadings_iteration[dim] / \
                    denominator[dim]
                # If  this dimension is the last in the tensor then update the eigvals
                # with the new loadings.
                if dim == len(tensor_dimensions) - 1:
                    eigvals[component] = norm(loadings_iteration[dim])
                #  update the loadings_iteration in dimension (dim)
                loadings_iteration[dim] = loadings_iteration[dim] / \
                    norm(loadings_iteration[dim])
                loadings[dim][:, component] = loadings_iteration[dim]
            # update the loadings with this iterations loadings
            for i, loading in enumerate(loadings):
                loading[:, component] = loadings_iteration[i]
        # MSE of the original tensor and the reconstructed tensor based on the
        # loadings.
        mean_squared_error = tensor - mask * \
            reconstruct_tensor(eigvals, loadings)
        # Frobenius norm for new reconstructed tensor
        iteration_tensor_frobenius_norm = norm(mean_squared_error)**2
        # If the error between this iterations reconstruction and the intital tensor is
        # below tol_als then  break the iterations.
        if np.sqrt(
                iteration_tensor_frobenius_norm /
                initial_tensor_frobenius_norm) < tol_als:
            break
    # get the final distance between the initial tensor and the reconstruction
    reconstructed_distance = np.sqrt(
        iteration_tensor_frobenius_norm /
        initial_tensor_frobenius_norm)
    # check that the factorization converged
    if any(sum(sum(np.isnan(loading))) > 0 for loading in loadings):
        raise ValueError("The factorization did not converge.",
                         "Please check the input tensor for errors.")
    # Get the diagonal of the eigvals
    eigvals = np.diag(eigvals.flatten())
    # Sort the eigvals and the laodings from
    # largest to smallest.
    idx = np.argsort(np.diag(eigvals))[::-1]
    eigvals = eigvals[idx, :][:, idx]
    loadings = [loading[:, idx] for loading in loadings]

    return loadings, eigvals, reconstructed_distance


def robust_tensor_power_method(
        tensor,
        n_components,
        n_initializations,
        max_rtpm_iterations,
        tol_rtpm):
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
    tensor : array-like
        A sparse `n` order tensor with zeros
        in place of missing values.
    r : int, optional
        The underlying low-rank, will be
        equal to the number of rank 1
        components that are output. The
        higher the rank given, the more
        expensive the computation will
        be.
    n_initializations : int, optional
        The number of initialization
        vectors. Larger values will
        give more accurate factorization
        but will be more computationally
        expensive.
    max_rtpm_iterations : int, optional
        Max number of iterations.
    tol_rtpm : flat, optional
        The minimization -- convergence break point for
        RTPM.

    Returns
    -------
    initial_eigvals : array-like
        The eigvals of the factorizations
    initial_loadings : list of array-like
        The `i`-th entry of initial_loadings corresponds to
        the factors along the `i`-th mode of tensor

    References
    ----------
    .. [2] A. Anandkumar, R. Ge, M., Janzamin,
            Guaranteed Non-Orthogonal Tensor
            Decomposition via Alternating Rank-1
            Updates. CoRR (2014),
            pp. 1-36.

    """
    # get the dimensions of the tensor for n loadings
    tensor_dimensions = tensor.shape
    # initialize loadings and eigvals as all zeros
    initial_loadings = [np.zeros((dim, n_components))
                        for dim in tensor_dimensions]
    initial_eigvals = np.zeros((n_components, 1))
    # begin power iterations for each rank-1 component
    for component in range(n_components):
        # initialize component loadings and eigvals as all zeros
        component_loadings = [np.zeros((dim, n_initializations))
                              for dim in tensor_dimensions]
        component_eigvals = np.zeros((n_initializations, 1))
        # for each initialization run a single iteration of RTPM
        for initialization in range(n_initializations):
            initializations = asymmetric_power_update(
                tensor -
                reconstruct_tensor(
                    initial_eigvals,
                    initial_loadings),
                max_rtpm_iterations=max_rtpm_iterations,
                tol_rtpm=tol_rtpm)
            # from this initialization generate the laodings
            for idx, initialization_loading in enumerate(component_loadings):
                initialization_loading[:,
                                       initialization] = initializations[idx]
                initialization_loading[:,
                                       initialization] = initialization_loading[:,
                                                                                initialization] / norm(initialization_loading[:,
                                                                                                                              initialization])
            # generate the eigvals
            component_eigvals[initialization] = tensor_distance(tensor - reconstruct_tensor(initial_eigvals, initial_loadings), [
                                                                initialization_loading[:, [initialization]] for initialization_loading in component_loadings])
        # find the best eigvals from all initializations and return that one
        idx = np.argmin(component_eigvals, axis=0)[0]
        # return the loadings and eigvals from that best initialization
        for initialization_loading, component_loading in zip(
                component_loadings, initial_loadings):
            component_loading[:,
                              component] = initialization_loading[:,
                                                                  idx] / norm(initialization_loading[:,
                                                                                                     idx])
        initial_eigvals[component] = tensor_distance(tensor - reconstruct_tensor(initial_eigvals, initial_loadings), [
                                                     component_loading[:, [component]] for component_loading in initial_loadings])

    return initial_eigvals, initial_loadings


def asymmetric_power_update(tensor, max_rtpm_iterations=50, tol_rtpm=1e-7):
    """
    Completes a single iteration of optimization
    for a random start of RTPM

    Parameters
    ----------
    tensor : array-like
        an `n` order tensor
    max_rtpm_iterations : int
        maximum iterations.

    Returns
    -------
    list of array-like
        entry `i` of list is a single
        vector corresponding to the `i`th
        mode of `tensor`

    """

    # RTPM_single
    n_dims = len(tensor.shape)
    # get tensor dimensions
    loadings = [randn(dim, 1) for dim in tensor.shape]
    loadings = [loading / norm(loading) for loading in loadings]
    for iteration in range(max_rtpm_iterations):
        # tensordot generalization to higher tensor_dimensions
        loadings_update = []
        tensor_dimensions = np.arange(n_dims)
        for dim in tensor_dimensions:
            dot_across = tensor_dimensions[tensor_dimensions != dim]
            dim_loadings_reconstructed = np.tensordot(
                tensor, loadings[dot_across[0]], axes=(1 if dim == 0 else 0, 0))
            for inner_dim in dot_across[1:]:
                dim_loadings_reconstructed = np.tensordot(
                    dim_loadings_reconstructed, loadings[inner_dim], axes=(
                        1 if inner_dim > dim else 0, 0))
            loadings_update.append(dim_loadings_reconstructed)

        new_shapes = [v_n.shape[:(-1 * (len(tensor_dimensions) - 2))]
                      for v_n in loadings_update]
        loadings_update = [
            v_n.reshape(new_shape) for v_n,
            new_shape in zip(
                loadings_update,
                new_shapes)]
        # save next iterations loadings and update to previous
        previous_loading = loadings
        loadings = [loading / norm(loading) for loading in loadings_update]
        # if the update converges then break from iterations
        if sum(
            norm(
                loading_prev -
                loading) for loading_prev,
            loading in zip(
                previous_loading,
                loadings)) < tol_rtpm:
            break

    return [loading.flatten() for loading in loadings]


def reconstruct_tensor(eigvals, initial_loadings):
    """
    This function takes the
    CP decomposition of a 3rd
    order tensor and outputs
    the reconstructed tensor
    reconstructed_tensor.

    Parameters
    ----------
    eigvals : array-like
        The r-dimension vector.
    initial_loadings : list of array-like
        Element i is a factor of shape
        (n[i], r).

    Returns
    -------
    T : array-like
        reconstructed_tensor of shape
        tuple(n[i] for i in range(len(initial_loadings))).
    """

    output_shape = tuple(loading.shape[0] for loading in initial_loadings)
    to_multiply = [
        eigvals.T *
        loading if i == 0 else loading for i,
        loading in enumerate(initial_loadings)]
    product = khatri_rao(to_multiply)
    T = product.sum(1).reshape(output_shape)

    return T


def tensor_distance(D, loadings):
    """
    The orthogonal tensor
    projection created by
    the tensor - reconstructed_tensor distance.
    Used in the initialization
    step with RTPM_single.

    Parameters
    ----------
    D : array-like
        with shape (n[0], n[1], ..., )
    loadings : list of array-like
        Element i is a factor of shape
        (n[i], r). Same length as D.shape

    Returns
    -------
    M : float
        The multilinear mapping of D on loadings
    """
    current = D
    for loading in loadings:
        current = np.tensordot(current, loading, axes=(0, 0))
    return current


def khatri_rao(matrices):
    """
    Returns the Khatri Rao product of a list of matrices

    Modified from TensorLy

    Parameters
    ----------
    matrices : list of array-like
        Matrices to take the Khatri Rao Product of

    Returns
    -------
    array-like
        The Khatri Rao Product of the matrices in `matrices`

    References
    ----------
    .. [1] Jean Kossaifi, Yannis Panagakis, Anima Anandkumar and Maja
            Pantic, TensorLy: Tensor Learning in Python,
            https://arxiv.org/abs/1610.09555.
    """

    n_columns = matrices[0].shape[1]
    n_factors = len(matrices)

    start = ord('a')
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(n_factors))
    source = ','.join(i + common_dim for i in target)
    operation = source + '->' + target + common_dim

    return np.einsum(operation, *matrices).reshape((-1, n_columns))
