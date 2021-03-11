# ----------------------------------------------------------------------------
# Copyright (c) 2019--, gemelli development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import warnings
import numpy as np
import pandas as pd
from scipy.linalg import svd
from numpy.linalg import norm
from .base import _BaseImpute
from scipy.spatial import distance
from gemelli.optspace import rank_estimate


class TensorFactorization(_BaseImpute):

    def __init__(self,
                 n_components=3,
                 max_als_iterations=50,
                 tol_als=1e-7,
                 max_rtpm_iterations=50,
                 n_initializations=50,
                 tol_rtpm=1e-5,
                 fillna=1.0,
                 center=True,
                 check_dense=True):
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
            The number of initial
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
        dist : array-like
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
        self.center = center
        self.check_dense = check_dense

    def fit(self, tensor, y=None):
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
        n_entries = np.product(tensor.shape)
        if self.check_dense:
            if (np.count_nonzero(tensor) == n_entries and
                    np.count_nonzero(~np.isnan(tensor)) == n_entries):
                err_ = 'No missing data in the format np.nan or 0.'
                raise ValueError(err_)
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
        loads, s, dist = tenals(tensor,
                                mask,
                                n_components=self.n_components,
                                n_initializations=self.n_initializations,
                                max_als_iterations=self.max_als_iterations,
                                max_rtpm_iterations=self.max_rtpm_iterations,
                                tol_als=self.tol_als,
                                tol_rtpm=self.tol_rtpm,
                                fillna=self.fillna)
        # save all raw laodings as attribute
        self.loadings = loads
        # the distance between tensor_imputed and tensor
        self.dist = dist
        # save array of loadings for subjects
        self.subjects = loads[0].copy()
        self.subjects = self.subjects[self.subjects[:, 0].argsort()]
        # save array of loadings for features
        self.features = loads[1].copy()
        self.features = self.features[self.features[:, 0].argsort()]
        # center the subject / feature biplot
        self.subjects -= self.subjects.mean(axis=0)
        self.features -= self.features.mean(axis=0)
        if self.center:
            # re-center using a final svd
            X = self.subjects @ s @ self.features.T
            possible_comp = [np.min(X.shape),
                             self.n_components]
            self.biplot_components = np.min(possible_comp)
            X = X - X.mean(axis=0)
            X = X - X.mean(axis=1).reshape(-1, 1)
            u, s, v = svd(X, full_matrices=False)
            u = u[:, :self.biplot_components]
            v = v.T[:, :self.biplot_components]
            p = s * (1 / s.sum())
            p = np.array(p[:self.biplot_components])
            s = np.diag(s[:self.biplot_components])
            # save the re-centered biplot
            self.features = v
            self.subjects = u
        else:
            # just make prop-exp
            p = np.array(np.diag(s) /
                         1 / np.sum(np.diag(s)))
        # save all eigen values
        self.eigvals = np.diag(s)
        # the proortion explained for n_components
        self.proportion_explained = p
        # save list of array(s) of loadings for conditions
        self.conditions = [loads[2]] if len(loads[2:]) == 1 \
            else loads[2:]
        # generate the trajectory(s) and distances
        # list of each condition-subject trajectory
        self.subject_trajectory = []
        # list of each condition-subject distance
        self.subject_distances = []
        # list of each condition-feature trajectory
        self.feature_trajectory = []
        # for each condition in conditions
        # generate a trajectory and distance array
        for condition in self.conditions:
            # temporary list of components in subject trajectory
            subject_temp_trajectory = []
            # temporary list of components in feature trajectory
            feature_temp_trajectory = []
            # for each component in the rank given to TensorFactorization
            for component in range(self.n_components)[::-1]:
                # component condition-subject trajectory
                dtmp = np.dot(loads[0][:, [component]],
                              condition[:, [component]].T).flatten()
                subject_temp_trajectory.append(dtmp)
                # component condition-feature trajectory
                dtmp = np.dot(loads[1][:, [component]],
                              condition[:, [component]].T).flatten()
                feature_temp_trajectory.append(dtmp)
            # combine all n_components
            subject_temp_trajectory = np.array(subject_temp_trajectory).T
            feature_temp_trajectory = np.array(feature_temp_trajectory).T
            # double check check centered
            subject_temp_trajectory -= subject_temp_trajectory.mean(axis=0)
            feature_temp_trajectory -= feature_temp_trajectory.mean(axis=0)
            # save subject-condition trajectory and distance matrix
            self.subject_trajectory.append(subject_temp_trajectory)
            self.subject_distances.append(
                distance.cdist(subject_temp_trajectory,
                               subject_temp_trajectory))
            # save feature-condition trajectory and distance matrix
            self.feature_trajectory.append(feature_temp_trajectory)

    def label(self,
              construct,
              taxonomy=None):
        """

        Label the loadings with the constructed tensor dimension
        labels from preprosessing.build.

        Parameters
        ----------
        construct : constructed object from preprosessing.build
            This object has the following attributes used here:
                construct.mapping : DataFrame
                    construct.mapping metadata used to build tensor
                    rows = samples
                    columns = categories
                construct.subject_order : list
                    order of subjects in tensor array
                construct.feature_order : list
                    order of features in tensor array
                construct.conditions : list of category names
                    category of conditional in metadata
                construct.condition_orders : list of lists
                    order of conditions for each
                    condition in tensor array
        taxonomy : DataFrame
            TODO
        """

        # columns labels PC1 ... PC(n_components)
        self.column_labels = ['PC' + str(i)
                              for i in range(1, self.n_components + 1)]
        self.biplot_labels = ['PC' + str(i)
                              for i in range(1, self.biplot_components + 1)]

        # % var explained
        self.proportion_explained = pd.Series(self.proportion_explained,
                                              index=self.biplot_labels)
        # eigvals
        self.eigvals = pd.Series(self.eigvals,
                                 index=self.biplot_labels)
        # DataFrame single non-condition dependent loadings
        self.subjects = pd.DataFrame(self.subjects,
                                     columns=self.biplot_labels,
                                     index=construct.subject_order)
        self.features = pd.DataFrame(self.features,
                                     columns=self.biplot_labels,
                                     index=construct.feature_order)

        # id taxonomy is given then add the taxonomic information
        if taxonomy is not None:
            self.features = pd.concat(
                [self.features, taxonomy], axis=1, sort=True)

        # list of DataFrame(s) for each condition loadings
        conditions = []
        for c_ind, condition in enumerate(construct.conditions):
            conditions.append(
                pd.DataFrame(
                    self.conditions[c_ind],
                    columns=self.column_labels,
                    index=construct.condition_orders[c_ind]))
        self.conditions = conditions

        # label the subject trajectories
        subject_trajectory = []
        for c_ind, condition in enumerate(construct.conditions):
            traj_tmp = self.subject_trajectory[c_ind]
            traj_tmp = pd.DataFrame(traj_tmp,
                                    columns=self.column_labels)
            ordr_ = [[subj, cond] for subj in construct.subject_order
                     for cond in construct.condition_orders[c_ind]]
            ordr_ = pd.DataFrame(ordr_, columns=['subject_id', condition])
            traj_tmp = pd.concat([traj_tmp, ordr_], axis=1)
            # add metadata for the trajectories
            index_map = construct.condition_metadata_map[c_ind]
            traj_tmp.index = [index_map[tuple(pair)]
                              if tuple(pair) in index_map.keys()
                              else '-'.join(list(map(str, pair)))
                              for pair in zip(traj_tmp['subject_id'].values,
                                              traj_tmp[condition].values)]
            traj_tmp.index.name = 'sample-id'
            subject_trajectory.append(traj_tmp)
        self.subject_trajectory = subject_trajectory

        # label the feature trajectories
        feature_trajectory = []
        for c_ind, condition in enumerate(construct.conditions):
            traj_tmp = self.feature_trajectory[c_ind]
            traj_tmp = pd.DataFrame(traj_tmp,
                                    columns=self.column_labels)
            ordr_ = [[feat, cond] for feat in construct.feature_order
                     for cond in construct.condition_orders[c_ind]]
            ordr_ = pd.DataFrame(ordr_, columns=['feature_id', condition])
            ordr_ = pd.concat([traj_tmp, ordr_], axis=1)
            if taxonomy is not None:
                if 'Taxon' in taxonomy.columns:
                    feat_map = dict(taxonomy.Taxon)
                    ordr_['Taxon'] = [feat_map[feat]
                                      if feat in feat_map.keys()
                                      else np.nan
                                      for feat in ordr_.feature_id]
                    # add taxonomic levels for grouping later (if available)

                    def tax_split(tax_id, tax_level): return tax_id.split(
                        tax_level)[1].split(';')[0]

                    for level, lname in zip(['k__', 'p__', 'c__', 'o__',
                                             'f__', 'g__', 's__'],
                                            ['kingdom', 'phylum', 'class',
                                             'order', 'family', 'genus',
                                             'species']):
                        if lname not in taxonomy.columns:
                            taxonomy_tmp = []
                            for tax in ordr_.Taxon:
                                if tax is not np.nan and\
                                   level in tax and\
                                   len(tax_split(tax, level)) > 0:
                                    taxonomy_tmp.append(tax_split(tax,
                                                                  level))
                                else:
                                    taxonomy_tmp.append(np.nan)
                            ordr_[lname] = taxonomy_tmp
                if 'Confidence' in taxonomy.columns:
                    feat_map = dict(taxonomy.Confidence)
                    ordr_['Confidence'] = [feat_map[feat]
                                           if feat in feat_map.keys()
                                           else np.nan
                                           for feat in ordr_.feature_id]

            ordr_.index.name = 'featureid'
            feature_trajectory.append(ordr_)
        self.feature_trajectory = feature_trajectory


def tenals(tensor,
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
        The number of initial
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
    dist : array-like
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
    # Frobenius norm initial for ALS minimization.
    initial_tensor_frobenius_norm = norm(tensor)**2
    # rank est. (for each slice)
    if len(tensor_dimensions) == 3:  # only for 3D
        for i in range(tensor_dimensions[0]):
            obs_tmp = tensor[i, :, :].copy()
            total_nonzeros = np.count_nonzero(mask[i, :, :].copy())
            n_, m_ = obs_tmp.shape
            eps_tmp = total_nonzeros / np.sqrt(n_ * m_)
            if min(obs_tmp.shape) <= 2:
                # two-subjects/time is already low-rank
                continue
            if rank_estimate(obs_tmp, eps_tmp) >= (min(obs_tmp.shape) - 1):
                warnings.warn('A component of your data may be high-rank.',
                              RuntimeWarning)
    # initial by Robust Tensor Power Method (modified for non-symmetric
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
            reconstructed_tensor = construct_tensor(eigvals, loadings)
            reconstructed_tensor = np.multiply(reconstructed_tensor, mask)
            # generate copy of loadings to reconstruct on each iterations
            loadings_iter = [loading[:, component].copy()
                             for loading in loadings]
            # denominator should end up as a list of np.zeros((dim_i, 1))
            denominator = [np.zeros(dim) for dim in tensor_dimensions]
            # for each  dimension of the tensor optimize that dimensions
            # loading based on the distance between the original tensor
            # and the reconstruction
            for dim, dim_size in enumerate(tensor_dimensions):
                # set previous loading to zero
                loadings[dim][:, component] = 0
                # generate indices to perform dot-product across
                dims_np = np.arange(len(tensor_dimensions))
                dot_across = dims_np[dims_np != dim]
                # outer dot-product
                err = tensor - reconstructed_tensor
                l_iter = loadings_iter[dot_across[0]]
                axes = (1 if dim == 0 else 0, 0)
                construct_loadings = np.tensordot(err,
                                                  l_iter,
                                                  axes=axes)
                denominator[dim] = np.tensordot(mask,
                                                l_iter**2,
                                                axes=axes)
                # inner dot-product
                for inner_dim in dot_across[1:]:
                    l_iter = loadings_iter[inner_dim]
                    axes = (1 if inner_dim > dim else 0, 0)
                    construct_loadings = np.tensordot(construct_loadings,
                                                      l_iter, axes=axes)
                    denominator[dim] = np.tensordot(denominator[dim],
                                                    l_iter**2, axes=axes)
                # update iteration's loadings
                loadings_iter[dim] = loadings[dim][:, component] +\
                    construct_loadings.flatten()
                # Add fillna to prevent division by zero in denominator
                # causing nan. This can occur in early iteration from all
                # zero fibers in the tensor along dim. In practice this
                # should be rare but can occur in very sparse tensors.
                denominator[dim][denominator[dim] == 0] = fillna
                loadings_iter[dim] = loadings_iter[dim] / \
                    denominator[dim]
                # If  this dimension is the last in the tensor then
                # update the eigvals with the new loadings.
                if dim == len(tensor_dimensions) - 1:
                    eigvals[component] = norm(loadings_iter[dim])
                #  update the loadings_iter in dimension (dim)
                eigvals[component][eigvals[component] == 0] = fillna
                loadings_iter[dim] = loadings_iter[dim] / \
                    eigvals[component]
                loadings[dim][:, component] = loadings_iter[dim]
            # update the loadings with this iterations loadings
            for i, loading in enumerate(loadings):
                loading[:, component] = loadings_iter[i]
        # MSE of the original tensor and the reconstructed tensor
        # based on the loadings.
        mean_squared_error = tensor - mask * \
            construct_tensor(eigvals, loadings)
        # Frobenius norm for new reconstructed tensor
        iteration_tensor_frobenius_norm = norm(mean_squared_error)**2
        # If the error between this iterations reconstruction and the
        # intital tensor is below tol_als then  break the iterations.
        err_conv = np.sqrt(iteration_tensor_frobenius_norm /
                           initial_tensor_frobenius_norm)
        if err_conv < tol_als:
            break
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
    loadings = [loading[:, idx[::-1]]
                for loading in loadings]

    return loadings, eigvals, err_conv


def robust_tensor_power_method(tensor,
                               n_components,
                               n_initializations=50,
                               max_rtpm_iterations=50,
                               tol_rtpm=1e-10,
                               random_state=42):
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
    n_components : int, optional
        The underlying low-rank, will be
        equal to the number of rank 1
        components that are output. The
        higher the rank given, the more
        expensive the computation will
        be.
    n_initializations : int, optional
        The number of initial
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
    eigvals : array-like
        The eigvals of the factorizations
    loadings : list of array-like
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

    # tensor shape is the number of loadings
    dims = tensor.shape
    # for each dim. initalize a loading fiber
    loadings = [np.ones((n, n_components)) for n in dims]
    # initlize n_component eigvals
    eigvals = np.ones((n_components, 1))
    # iterate on loadings from first to last
    for r in range(n_components):
        # loadings and eigs to fille for
        # each random initalization vector
        tU = [np.zeros((n, n_initializations)) for n in dims]
        tS = np.zeros((n_initializations, 1))
        # random initialization
        if random_state is None or isinstance(random_state, int):
            rnd = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            rnd = random_state
        else:
            raise ValueError('Random State must be of type ',
                             'np.random.RandomState or int.')
        # run n-initializations
        for init in range(n_initializations):
            # single random initialization
            init_load = [rnd.random_sample((n, 1)) for n in dims]
            init_load = [vec / norm(vec) for vec in init_load]
            # construct tensor
            T_con = construct_tensor(eigvals, init_load)
            # run power iteration's for loading initialization
            initializations = asymmetric_power_update(tensor - T_con,
                                                      init_load,
                                                      max_rtpm_iterations,
                                                      tol_rtpm)
            # fill that initalization (looking for maximum)
            for idx, load_tmp in enumerate(initializations):
                tU[idx][:, init] = load_tmp
                tU[idx][:, init] = tU[idx][:, init] / norm(tU[idx][:, init])
            # generate eigvals for that initalization
            T_con = construct_tensor(eigvals, loadings)
            tS[init] = eigval_update(tensor - T_con,
                                     [ltmp[:, [init]] for ltmp in tU])
        # find the maximum (absolute) eigval
        max_idx = np.argmax(abs(tS))
        # for that optimal eigval add it to
        # the final laodings & eigvals.
        for idx, max_load in enumerate(tU):
            loadings[idx][:, r] = max_load[:, max_idx]
            loadings[idx][:, r] = (loadings[idx][:, r]
                                   / norm(loadings[idx][:, r]))
        # fill eigval
        T_con = construct_tensor(eigvals, loadings)
        eigvals[r] = eigval_update(tensor - T_con,
                                   [ltmp[:, [r]] for ltmp in loadings])

    return eigvals, loadings


def asymmetric_power_update(tensor,
                            init,
                            max_rtpm_iterations=50,
                            tol_rtpm=1e-10):
    """
    Completes a single iteration of optimization
    for a random start of RTPM

    Parameters
    ----------
    tensor : array-like
        an `n` order tensor
    init : list of array-like
        randomly generated loadings
        for each dimension of tensor
    max_rtpm_iterations : int
        maximum iterations.

    Returns
    -------
    list of array-like
        entry `i` of list is a single
        vector corresponding to the `i`th
        mode of `tensor`

    """

    # tensor dims is number of loadings
    n_dims = len(tensor.shape)
    # begin power iterations
    for itr in range(max_rtpm_iterations):
        loadings = []
        dims = np.arange(n_dims)
        # iterate on each loading (dim.)
        for dim in dims:
            # generate updates for each loading dimention
            dot_across = dims[dims != dim]
            axis = (1 if dim == 0 else 0, 0)
            load_dim = np.tensordot(tensor,
                                    init[dot_across[0]],
                                    axes=axis)
            for inner_dim in dot_across[1:]:
                axis = (1 if inner_dim > dim else 0, 0)
                load_dim = np.tensordot(load_dim,
                                        init[inner_dim],
                                        axes=axis)
            loadings.append(load_dim)
        # fill the update loadings
        new_shapes = [u_n.shape[:(-1 * (len(dims) - 2))]
                      for u_n in loadings]
        loadings = [u_n.reshape(new_shape)
                    for u_n, new_shape in zip(loadings,
                                              new_shapes)]
        # update initalized loading and keep prev.
        init_prev = [u for u in init]
        init = [u_i / norm(u_i) for u_i in loadings]
        # calulate the iteration minimization (exit case for tol_rtpm)
        tol_itr = sum(norm(u0 - u) for u0, u in zip(init_prev, init))
        if tol_itr < tol_rtpm:
            break
    return [loading.flatten() for loading in init]


def construct_tensor(S, U):
    """
    This function takes the
    CP decomposition of a 3rd
    order tensor and outputs
    the reconstructed tensor
    TE_hat.
    Parameters
    ----------
    S : array-like
        The r-dimension vector.
    U : list of array-like
        Element i is a factor of shape
        (n[i], r).
    Returns
    -------
    T : array-like
        TE_hat of shape
        tuple(n[i] for i in range(len(U))).
    """

    output_shape = tuple(u.shape[0] for u in U)
    to_multiply = [S.T * u if i == 0 else u for i, u in enumerate(U)]
    product = khatri_rao(to_multiply)
    T = product.sum(1).reshape(output_shape)

    return T


def eigval_update(D, U_list):
    """
    The Orthogonal tensor
    projection created by
    the TE - TE_hat distance.
    Used in the initialization
    step with RTPM_single.
    Parameters
    ----------
    D : array-like
        with shape (n[0], n[1], ..., )
    U_list : list of array-like
        Element i is a factor of shape
        (n[i], r). Same length as D.shape
    Returns
    -------
    M : float
        The multilinear mapping of D on U_list
    """
    current = D
    for u in U_list:
        current = np.tensordot(current, u, axes=(0, 0))
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
