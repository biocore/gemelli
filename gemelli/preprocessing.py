# ----------------------------------------------------------------------------
# Copyright (c) 2019--, gemelli development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import warnings
import numpy as np
from biom import Table
from .base import _BaseConstruct


def tensor_rclr(T):
    """
    Robust clr transform. is the approximate geometric mean of X.

    We know from the Central Limit Theorem that as
    we collect more independent measurements we
    approach the true geometric mean.

    This transformation will work on N mode tensors
    by flattening. Flattened tensor are reshaped into
    subject x contions by features before transformation.
    A tensor will be returned in the shape shape and order.

    Mode 2 tensors (matrix) will be directly transformed,
    no reshaping necessary.

    Parameters
    ----------
    T : array-like
        Array of non-negative count data.
        In an N mode tensor of shape:
        first dimension = samples
        second dimension = features
        [3..N] dimensions = conditions

    Raises
    ------
    ValueError
        Tensor is less than 2-dimensions.
    ValueError
        Tensor contains negative values.
    ValueError
        Tensor contains np.inf or -np.inf.
    ValueError
        Tensor contains np.nan or missing.

    References
    ----------
    .. [1] V. Pawlowsky-Glahn, J. J. Egozcue,
           R. Tolosana-Delgado (2015),
           Modeling and Analysis of
           Compositional Data, Wiley,
           Chichester, UK

    .. [2] C. Martino et al., A Novel Sparse
           Compositional Technique Reveals
           Microbial Perturbations. mSystems.
           4 (2019), doi:10.1128/mSystems.00016-19.

    Examples
    --------
    TODO

    """

    if len(T.shape) < 2:
        raise ValueError('Tensor is less than 2-dimensions')

    if np.count_nonzero(np.isinf(T)) != 0:
        raise ValueError('Tensor contains either np.inf or -np.inf.')

    if np.count_nonzero(np.isnan(T)) != 0:
        raise ValueError('Tensor contains np.nan or missing.')

    if (T < 0).any():
        raise ValueError('Tensor contains negative values.')

    if len(T.shape) < 3:
        # tensor_rclr on 2D matrix
        M_tensor_rclr = matrix_rclr(T.transpose().copy()).T
        M_tensor_rclr[~np.isfinite(M_tensor_rclr)] = 0.0
        return M_tensor_rclr
    else:
        # flatten tensor (samples*conditions x features)
        T = T.copy()
        # conditional dimensions
        conditions_index = list(range(2, len(T.shape)))
        forward_T = tuple([0] + conditions_index + [1])
        reverse_T = tuple([0] + [conditions_index[-1]]
                          + [1] + conditions_index[:-1])
        # transpose to flatten
        T = T.transpose(forward_T)
        M = T.reshape(np.product(T.shape[:len(T.shape) - 1]),
                      T.shape[-1])
        with np.errstate(divide='ignore', invalid='ignore'):
            M_tensor_rclr = matrix_rclr(M)
        M_tensor_rclr[~np.isfinite(M_tensor_rclr)] = 0.0
        # reshape to former tensor and return tensors
        return M_tensor_rclr.reshape(T.shape).transpose(reverse_T)


def matrix_rclr(M):
    """
    Robust clr transform helper function.
    This function is built for mode 2 tensors,
    also known as matrices.

    Raises
    ------
    ValueError
       Raises an error if any values are negative.
    ValueError
       Raises an error if the matrix has more than 2 dimension.

    References
    ----------
    .. [1] V. Pawlowsky-Glahn, J. J. Egozcue,
           R. Tolosana-Delgado (2015),
           Modeling and Analysis of
           Compositional Data, Wiley,
           Chichester, UK

    .. [2] C. Martino et al., A Novel Sparse
           Compositional Technique Reveals
           Microbial Perturbations. mSystems.
           4 (2019), doi:10.1128/mSystems.00016-19.

    Examples
    --------
    TODO

    """
    # ensure array is at least 2D
    M = np.atleast_2d(np.array(M))
    # ensure array not more than 2D
    if M.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    # ensure no neg values
    if (M < 0).any():
        raise ValueError('Array Contains Negative Values')
    # ensure no undefined values
    if np.count_nonzero(np.isinf(M)) != 0:
        raise ValueError('Data-matrix contains either np.inf or -np.inf')
    # ensure no missing values
    if np.count_nonzero(np.isnan(M)) != 0:
        raise ValueError('Data-matrix contains nans')
    # closure following procedure in
    # skbio.stats.composition.closure
    M_log = M / M.sum(axis=1, keepdims=True)
    # log transform before geo-mean
    M_log = np.log(M_log.squeeze())
    mask = [True] * np.product(M_log.shape)
    mask = np.array(mask).reshape(M_log.shape)
    mask[np.isfinite(M_log)] = False
    # sum of rows (features)
    M_tensor_rclr = np.ma.array(M_log, mask=mask)
    # approx. geometric mean of the features
    gm = M_tensor_rclr.mean(axis=-1, keepdims=True)
    # subtracted to center log
    M_tensor_rclr = (M_tensor_rclr - gm).squeeze().data
    # ensure any missing are zero again
    M_tensor_rclr[~np.isfinite(M_log)] = np.nan
    return M_tensor_rclr


def rclr_transformation(table: Table) -> Table:
    """
    Takes biom table and returns
    a matrix_rclr transformed biom table.
    """
    # transform table values (and return biom.Table)
    table = Table(matrix_rclr(table.matrix_data.toarray().T).T,
                  table.ids('observation'),
                  table.ids('sample'))
    return table


class build(_BaseConstruct):

    """
    This class can both build N-mode
    tensors from 2D dataframes given
    a count table and mapping data.

    A list of conditional measurements are
    given that identify context measured
    multiple times over the same subjects.
    Additionally a set of subject IDs
    must be provided. Any subjects that are
    missing in a given condition are left
    as completely zero.

    Parameters
    ----------
    table : DataFrame
        table of non-negative count data
        rows = features
        columns = samples
    mapping : DataFrame
        mapping metadata for table
        rows = samples
        columns = categories
    subjects : str, int, or float
        category of sample IDs in metadata
    conditions : str, int, or float
        category of conditional in metadata

    Attributes
    -------
    subject_order : list
        order of subjects in tensor array
    feature_order : list
        order of features in tensor array
    condition_orders : list of lists
        order of conditions for each
        condition in tensor array
    counts : array-like
        Contains table counts.
        N mode tensor of shape
        first dimension = samples
        second dimension = features
        [3..N] dimensions = conditions

    Raises
    ------
    ValueError
        if subject not in mapping
    ValueError
        if any conditions not in mapping
    ValueError
        Table is not 2-dimensional
    ValueError
        Table contains negative values
    ValueError
        Table contains np.inf or -np.inf
    ValueError
        Table contains np.nan or missing.
    Warning
        If a conditional-sample pair
        has multiple IDs associated
        with it the multiple samples
        are meaned.

    References
    ----------
    .. [1] V. Pawlowsky-Glahn, J. J. Egozcue, R. Tolosana-Delgado (2015),
    Modeling and Analysis of Compositional Data, Wiley, Chichester, UK

    .. [2] C. Martino et al., A Novel Sparse Compositional Technique Reveals
    Microbial Perturbations. mSystems. 4 (2019), doi:10.1128/mSystems.00016-19.

    Examples
    --------
    TODO

    """

    def __init__(self):
        """
        Parameters
        ----------
        None

        """

    def construct(self, table, mf, subjects, conditions):
        """
        This function transforms a 2D table
        into a N-Order tensor.

        Parameters
        ----------
        table : DataFrame
            table of non-negative count data
            rows = features
            columns = samples
        mapping : DataFrame
            mapping metadata for table
            rows = samples
            columns = categories
        subjects : str, int, or float
            category of sample IDs in metadata
        conditions : list of strings or ints
            category of conditional in metadata

        Returns
        -------
        self to abstract method

        Raises
        ------
        ValueError
            if subject not in mapping
        ValueError
            if any conditions not in mapping
        ValueError
            Table is not 2-dimensional
        ValueError
            Table contains negative values
        ValueError
            Table contains np.inf or -np.inf
        ValueError
            Table contains np.nan or missing.
        Warning
            If a conditional-sample pair
            has multiple IDs associated
            with it. In this case the
            default method is to mean them.

        Examples
        --------
        TODO

        """

        if subjects not in mf.columns:
            raise ValueError("Subject provided (" +
                             str(subjects) +
                             ") category not in metadata columns.")

        if any(cond_col not in mf.columns for cond_col in conditions):
            missin_cond = ','.join([cond_col for cond_col in conditions
                                    if cond_col not in mf.columns])
            raise ValueError("Conditional category(s) [" +
                             str(missin_cond) +
                             "] not in metadata column(s).")

        if np.count_nonzero(np.isinf(table.values)) != 0:
            raise ValueError('Table contains either np.inf or -np.inf.')

        if np.count_nonzero(np.isnan(table.values)) != 0:
            raise ValueError('Table contains np.nan or missing.')

        if (table.values < 0).any():
            raise ValueError('Table contains negative values.')

        # store all to self
        self.table = table.copy()
        self.mf = mf.copy()
        self.subjects = subjects
        self.conditions = conditions
        self._construct()

        return self

    def _construct(self):
        """
        This function forms a tensor
        with missing subject x condition
        pairs left as all zeros.

        Raises
        ------
        Warning
            If a conditional-subject pair
            has multiple samples associated
            with it. In this case the
            default method is to mean them.

        """

        table, mf = self.table, self.mf

        # Step 1: mean samples with multiple conditional overlaps
        col_tmp = [self.subjects] + self.conditions
        duplicated = {k: list(df.index)
                      for k, df in mf.groupby(col_tmp)
                      if df.shape[0] > 1}  # get duplicated conditionals
        if len(duplicated.keys()) > 0:
            duplicated_ids = ','.join(list(set([str(k[0])
                                                for k in duplicated.keys()])))
            warnings.warn(''.join(["Subject(s) (", str(duplicated_ids),
                                   ") contains multiple ",
                                   "samples. Multiple subject counts will be",
                                   " meaned across samples by subject."]),
                          RuntimeWarning)
        for id_, dup in duplicated.items():
            # mean and keep one
            table[dup[0]] = table.loc[:, dup].mean(axis=1).astype(int)
            # drop the other
            table.drop(dup[1:], axis=1)
            mf.drop(dup[1:], axis=0)
        # save direct data
        table_counts = table.values

        # Step 2: fill the tensor (missing are all zero)

        # generate all sorted mode ids
        def sortset(ids): return sorted(set(ids))
        # get the ordered subjects
        subject_order = sortset(mf[self.subjects])
        # get un-ordered features (order does not matter)
        feature_order = list(table.index)
        # get the ordered for each conditional
        conditional_orders = [sortset(mf[cond])
                              for cond in self.conditions]
        # generate the dims.
        all_dim = [subject_order]\
            + conditional_orders  # concat all

        # get tensor to fill with counts (all zeros)
        shape = tuple([len(cl) for cl in [subject_order,
                                          feature_order]
                       + all_dim[1:]])
        tensor_counts = np.zeros(tuple(shape))

        # generate map from ordered subject and conditions
        # to the original orders in the table
        projection = {
            tuple(
                dim.index(k_) for k_, dim in zip(
                    k, all_dim)): list(
                table.columns).index(
                    df.index[0]) for k, df in mf.groupby(
                        [
                            self.subjects] + self.conditions)}

        # fill the tensor with data
        for T_ind, M_ind in projection.items():
            # get the index from the tensor
            ind_ = [T_ind[:1], list(range(len(table.index)))] + list(T_ind[1:])
            # fill count tensor from table
            tensor_counts[tuple(ind_)] = table_counts[:, M_ind]

        # save metadat and save subject-conditional index
        condition_metadata_map = [{(sid, con): i
                                  for i, sid, con in zip(mf.index,
                                                         mf[self.subjects],
                                                         mf[con])}
                                  for con in self.conditions]
        self.condition_metadata_map = condition_metadata_map
        # save tensor label order
        self.counts = tensor_counts
        self.subject_order = subject_order
        self.feature_order = feature_order
        self.condition_orders = conditional_orders
        self.mf = self.mf
