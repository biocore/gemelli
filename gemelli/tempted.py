import copy
import numpy as np
import pandas as pd
import warnings
from scipy.sparse.linalg import svds
from gemelli.optspace import svd_sort
np.random.seed(42)


def freg_rkhs(Ly, a_hat, ind_vec, Kmat,
              Kmat_output, smooth=1e-6):

    """
    A helper function for tempted
    to update phi (state loadings).

    Parameters
    ----------
    Ly: list of numpy.ndarray
        list of length equal to
        the number of individuals.
        Each item is an array of
        size equal to the number
        of states for that individual.

    a_hat: numpy.ndarray
        The rank-one individual loadings.
        rows = individual
        columns = None

    ind_vec: numpy.ndarray
        Indexing of the samples
        associated with an individual.
        rows = N samples
        columns = None

    Kmat: numpy.ndarray
        The kernel matrix.
        rows = samples
        columns = samples

    Kmat_output: numpy.ndarray
        The bernoulli kernel matrix.
        rows = states (resolution)
        columns = samples

    smooth: float
        1e-6

    Returns
    -------
    numpy.ndarray
        Updated rank-1 phi
        rows = states (resolution)
        columns = None

    """

    A = Kmat.copy()
    for i in range(len(Ly)):
        A[ind_vec == i, :] *= (a_hat[i] ** 2)
    cvec = np.hstack(Ly)
    A_temp = A + smooth * np.eye(A.shape[1])
    beta = np.linalg.inv(A_temp) @ cvec
    phi_est = Kmat_output.dot(beta)
    return phi_est


def bernoulli_kernel(x, y):

    """
    This function is used to calculate the
    kernel matrix for the RKHS regression
    that iteratively updates the temporal
    loading function.

    Parameters
    ----------
    x: numpy.ndarray
        rows = states (resolution)
        columns = None

    y: numpy.ndarray
        rows = states (resolution)
        columns = None

    Returns
    -------
    numpy.ndarray
        rows = states (resolution)
        columns = states (resolution)

    """

    k1_x = x - 0.5
    k1_y = y - 0.5
    k2_x = 0.5 * (k1_x ** 2 - 1 / 12)
    k2_y = 0.5 * (k1_y ** 2 - 1 / 12)
    xy = np.abs(np.outer(x, np.ones_like(y)) - np.outer(np.ones_like(x), y))
    k4_xy = 1 / 24 * ((xy - 0.5) ** 4 - 0.5 * (xy - 0.5) ** 2 + 7 / 240)
    kern_xy = np.outer(k1_x, k1_y) + np.outer(k2_x, k2_y) - k4_xy + 1

    return kern_xy


def tempted(individual_id_tables,
            individual_id_state_orders,
            feature_order,
            n_components=3, smooth=1e-6,
            interval=None, resolution=101,
            maxiter=20, epsilon=1e-4):

    """
    Decomposition of temporal tensors.

    Parameters
    ----------
    individual_id_tables: dictionary, required
        Dictionary of tables constructed
        (see build_sparse class).
        keys = individual_ids
        values = DataFrame
            rows = features
            columns = samples

    individual_id_state_orders: dictionary, required
        Dictionary of subjects time points
        for each sample.
        (see build_sparse class).
        keys = individual_ids
        values = numpy.ndarray
            rows = states
            columns = None

    feature_order: list, required
        The order of the features used
        to map the tables (see build_sparse class).

    n_components: int, optional : Default is 3
        The underlying rank of the data and number of
        output dimentions.

    smooth: float, optional : Default is 1e-8
        Smoothing parameter for RKHS norm. Larger means
        smoother temporal loading functions.

    interval: range of type float, optional : Default is None
        The range of time points to ran the decomposition for.

    resolution: int, optional : Default is 101
        Number of time points to evaluate the value
        of the temporal loading function.

    maxiter: int, optional : Default is 20
        Maximum number of iteration in for rank-1 calculation.

    epsilon: float, optional : Default is 0.0001
        Convergence criteria for difference between iterations
        for each rank-1 calculation.

    Returns
    -------
    DataFrame
        The individual loadings.
        rows = individual IDs
        columns = components

    DataFrame
        The feature loadings.
        rows = features
        columns = components

    DataFrame
        The state loadings.
        rows = states (N = resolution)
        columns = components

    DataFrame
        The mapping from the integer time
        and time points where the temporal
        loading function is evaluated.

    numpy.ndarray
        Eigen values, for each component.

    numpy.ndarray
        Variance explained by each component.

    Raises
    ------
    ValueError
        if features don't match between tables
        across the values of the dictionary

    Examples
    --------
    # load tables
    table = load_table(in_table_path)
    sample_metadata = read_csv(in_metadata_path,
                               sep='\t',
                               index_col=0)
    # tensor building
    sparse_tensor = build_sparse()
    sparse_tensor.construct(table,
                            sample_metadata,
                            'host_subject_id',
                            'time_points')
    # run TEMPTED
    tempted_res = tempted(sparse_tensor.individual_id_tables_centralized,
                          sparse_tensor.individual_id_state_orders,
                          sparse_tensor.feature_order, resolution=10)

    """

    # make copy of tables to update
    tables_update = copy.deepcopy(individual_id_tables)
    orders_update = copy.deepcopy(individual_id_state_orders)
    # get number of individuals
    n_individuals = len(tables_update)
    # get number of features and check all tables are the same
    n_features_all = [m.shape[0] for m in tables_update.values()]
    if not all([v == n_features_all[0] for v in n_features_all]):
        raise ValueError('Individual tables do not'
                         ' have the same number of features.')
    if not all([all(feature_order == m.index)
                for m in tables_update.values()]):
        raise ValueError('Individual tables do not'
                         ' all have the same features.')
    n_features = n_features_all[0]
    # init dataframes to fill
    lambda_coeff = np.zeros(n_components)
    rsquared = np.zeros(n_components)
    n_component_col_names = ['component_' + str(i+1)
                             for i in range(n_components)]
    individual_loadings = pd.DataFrame(np.zeros((n_individuals, n_components)),
                                       index=tables_update.keys(),
                                       columns=n_component_col_names)
    feature_loadings = pd.DataFrame(np.zeros((n_features, n_components)),
                                    index=feature_order,
                                    columns=n_component_col_names)
    state_loadings = pd.DataFrame(np.zeros((resolution, n_components)),
                                  columns=n_component_col_names)
    # set the interval if none is given
    timestamps_all = np.concatenate(list(orders_update.values()))
    timestamps_all = np.unique(timestamps_all)
    if interval is None:
        interval = (timestamps_all[0], timestamps_all[-1])
    # set time ranges [0, 1]
    input_time_range = (timestamps_all[0], timestamps_all[-1])
    for individual_id in orders_update.keys():
        orders_update[individual_id] = (orders_update[individual_id]
                                        - input_time_range[0]) \
                                        / (input_time_range[1]
                                           - input_time_range[0])
    interval = tuple((interval
                      - input_time_range[0])
                     / (input_time_range[1]
                        - input_time_range[0]))
    # set resolution mask
    ti = [[] for i in range(n_individuals)]
    tipos = []
    Lt = []
    ind_vec = []
    y0 = []
    for i, (id_, time_range_i) in enumerate(orders_update.items()):
        temp = 1 + np.array(list(map(int, (resolution-1) * (time_range_i -
                                                            interval[0]) /
                            (interval[1] - interval[0]))))
        temp[(temp <= 0) | (temp > resolution)] = 0
        ti[i] = temp - 1
        keep = ti[i] >= 0
        tipos.append(keep)
        Lt.append(time_range_i)
        ind_vec.extend([i] * len(Lt[-1]))
        # get inital matrix sets to compare later
        y0.append(tables_update[id_].T[tipos[i]].values)
    ind_vec = np.array(ind_vec)
    y0 = np.concatenate(y0).T.flatten()
    # kernel matrix calculatioin
    tm = np.concatenate(Lt)
    Kmat = bernoulli_kernel(tm, tm)
    Kmat_output = bernoulli_kernel(np.linspace(interval[0],
                                               interval[1],
                                               num=resolution),
                                   tm)
    # calculate rank-1 component sequentially.
    X = []
    for s in range(n_components):
        # initialization of feature loadings
        data_unfold = np.hstack([m.values for m in tables_update.values()])
        u, e, v = svds(data_unfold, k=n_components, which='LM')
        u, e, v = svd_sort(u, np.diag(e), v)
        b_hat = u[:, 0]
        consistent_sign = np.sign(np.sum(b_hat))
        # initialization of subject loadings
        a_hat = (np.ones(n_individuals)
                 / np.sqrt(n_individuals)) * consistent_sign
        # iteratively update subject, feature, & state loadings
        t = 0
        dif = 1
        while t <= maxiter and dif > epsilon:
            # update state loadings
            Ly = [a_hat[i] * b_hat.dot(m)
                  for i, m in enumerate(tables_update.values())]
            phi_hat = freg_rkhs(Ly, a_hat, ind_vec,
                                Kmat, Kmat_output,
                                smooth=smooth)
            phi_hat = (phi_hat / np.sqrt(np.sum(phi_hat ** 2)))
            # update subject & feature loadings
            a_tilde = np.zeros(n_individuals)
            temp_num = np.zeros((n_features, n_individuals))
            temp_denom = np.zeros(n_individuals)
            for i, m in enumerate(tables_update.values()):
                t_temp = tipos[i]
                phi_ = phi_hat[ti[i][t_temp]]
                a_tilde[i] = b_hat.dot(m.T[t_temp].values.T).dot(phi_)
                a_tilde[i] = a_tilde[i] / np.sum(phi_ ** 2)
                temp_num[:, i] = (m.T[t_temp].values.T).dot(phi_)
                temp_denom[i] = np.sum(phi_ ** 2)
            # update subject
            a_new = a_tilde / np.sqrt(np.sum(a_tilde ** 2))
            dif = np.sum((a_hat - a_new) ** 2)
            a_hat = a_new
            # update feature loadings
            b_tilde = temp_num.dot(a_hat) / (temp_denom.dot(a_hat ** 2))
            b_new = b_tilde / np.sqrt(np.sum(b_tilde ** 2))
            dif = max(dif, np.sum((b_hat - b_new) ** 2))
            b_hat = b_new
            t += 1
        # save rank-1 minimization
        x = []
        y = []
        for i, m in enumerate(tables_update.values()):
            x.append(a_hat[i] * np.outer(b_hat, phi_hat[ti[i][ti[i] >= 0]]))
            y.append(m.T[tipos[i]].values)
        x = np.concatenate(x, axis=1)
        y = np.concatenate(y)
        X.append(x.flatten())
        x = x.flatten().reshape(-1, 1)
        # get coeff and r2
        lambda_coeff_, resid = np.linalg.lstsq(x, y.T.flatten(),
                                               rcond=-1)[:2]
        individual_loadings.iloc[:, s] = a_hat
        feature_loadings.iloc[:, s] = b_hat
        state_loadings.iloc[:, s] = phi_hat.flatten()
        lambda_coeff[s] = lambda_coeff_
        rsquared[s] = 1 - resid / (y.size * y.var())
        # update data
        for i, (individual_id, m) in enumerate(tables_update.items()):
            temp = tipos[i]
            ft_tmp_ = feature_loadings.iloc[:, [s]]
            st_tmp_ = state_loadings.iloc[ti[i][temp], [s]]
            scale_tmp = ft_tmp_.values.dot(st_tmp_.values.T)
            scale_tmp = individual_loadings.iloc[i, [s]].values * scale_tmp
            tables_update[individual_id] -= (lambda_coeff[[s]] * scale_tmp)
    # accum lambdas
    X = np.vstack(X).T
    # get coeff and r2
    lambda_coeff_ = np.linalg.lstsq(X, y0, rcond=-1)[0]
    # revise the sign of lambda_coeff
    lambda_coeff = np.where(lambda_coeff < 0, -lambda_coeff, lambda_coeff)
    individual_loadings = pd.DataFrame(np.where(lambda_coeff[:,
                                                             np.newaxis].T < 0,
                                                -individual_loadings,
                                                individual_loadings),
                                       individual_loadings.index,
                                       individual_loadings.columns)
    # revise the signs to make sure summation is nonnegative
    sgn_state_loadings = np.sign(state_loadings.sum(axis=0))
    state_loadings *= sgn_state_loadings
    # revise the signs to make sure summation is nonnegative
    sgn_feature_loadings = np.sign(feature_loadings.sum(axis=0))
    feature_loadings *= sgn_feature_loadings.values
    individual_loadings *= sgn_feature_loadings.values
    # return new time scale from input resolution
    time_return = np.linspace(interval[0],
                              interval[1],
                              resolution)
    time_return *= (input_time_range[1] - input_time_range[0])
    time_return += input_time_range[0]
    time_return = pd.DataFrame(time_return,
                               state_loadings.index,
                               ['time_interval'])

    return (individual_loadings, feature_loadings,
            state_loadings, time_return,
            lambda_coeff, rsquared)


def tempted_transform(tables_test,
                      state_orders_test,
                      fl_train,
                      state_loading_train,
                      eigen_coeff_train,
                      time_train,
                      v_centralized_train=None,
                      subset_tables=True):

    """
    This function estimates the subject loading of the testing data
    based on feature and temporal loading from training data.

    Parameters
    ----------
    tables_test: dictionary, required
        Dictionary of tables constructed
        (see build_sparse class).
        keys = individual_ids
        values = DataFrame
            rows = features
            columns = samples

    state_orders_test: dictionary, required
        Dictionary of subjects time points
        for each sample.
        (see build_sparse class).
        keys = individual_ids
        values = numpy.ndarray
            rows = states
            columns = None

    fl_train: DataFrame, required
        The feature loadings.
        rows = features
        columns = components

    state_loading_train: DataFrame, required
        The state loadings.
        rows = states (N = resolution)
        columns = components

    eigen_coeff_train: numpy.ndarray, required
        Eigen values, for each component.

    time_train: DataFrame, required
        The mapping from the integer time
        and time points where the temporal
        loading function is evaluated.

    v_centralized_train: numpy.ndarray, optional: None
        V from svd approximation.

    subset_tables: bool, optional: True
        Subsets the input tables to contain only features used in the
        training data. If set to False and the tables are not perfectly
        matched a ValueError will be produced.

    Returns
    -------
    DataFrame
        The projected individual loadings.
        rows = individual IDs
        columns = components


    Raises
    ------
    ValueError
        `ValueError: The input tables do not contain all
        the features in the ordination.`.

    ValueError
        `ValueError: Removing # features(s) in table(s)
        but not the ordination.`.

    ValueError
        `ValueError: Features in the input table(s) not in
        the features in the ordination.  Either set subset_tables to
        True or match the tables to the ordination.`.


    Examples
    --------
    # load tables train
    table_train = load_table(in_table_path)
    sample_metadata_train = read_csv(in_metadata_path,
                                     sep='\t',
                                     index_col=0)
    # tensor building train
    st_train = build_sparse()
    st_train.construct(table_train,
                                  sample_metadata_train,
                                  'host_subject_id',
                                  'time_points')
    # run TEMPTED train
    tempted_res_train = tempted(st_train.individual_id_tables_centralized,
                                st_train.individual_id_state_orders,
                                st_train.feature_order, resolution=10)

    # load tables train
    table_test = load_table(in_table_path_test)
    sample_metadata_test = read_csv(in_metadata_path_test,
                                    sep='\t',
                                    index_col=0)
    # tensor building train
    st_test = build_sparse()
    st_test.construct(table_test,
                                 sample_metadata_test,
                                 'host_subject_id',
                                 'time_points')
    # project new data
    vc_train = st_train.v_centralized.copy()
    proj_res = tempted_transform(st_test.individual_id_tables.copy(),
                                 st_test.individual_id_state_orders,
                                 tempted_res_train[1].copy(),
                                 tempted_res_train[2].copy(),
                                 tempted_res_train[4].copy(),
                                 tempted_res_train[3].copy(),
                                 v_centralized_train = vc_train)

    """

    # first run check to make sure it is possible
    # ensure feature IDs match
    test_features_order = list(list(tables_test.values())[0].index)
    shared_features = set(test_features_order) & set(fl_train.index)
    if len(shared_features) < len(set(fl_train.index)):
        raise ValueError('The input tables do not contain all'
                         ' the features in the ordination.')
    elif subset_tables:
        unshared_N = len(set(test_features_order)) - len(shared_features)
        if unshared_N != 0:
            warnings.warn('Removing %i features(s) in table(s)'
                          ' but not the ordination.'
                          % (unshared_N), RuntimeWarning)
        tables_test = {id_: m.reindex(fl_train.index)
                       for id_, m in tables_test.items()}
    else:
        raise ValueError('Features in the input table(s) not in'
                         ' the features in the ordination.'
                         ' Either set subset_tables to True or'
                         ' match the tables to the ordination.')
    # convert back to the array representation
    time_train = time_train.values.flatten()
    n_components = len(fl_train.columns)
    n_individuals = len(tables_test)
    n_features = len(fl_train.index)
    # Initialize the output matrix.
    id_projected = np.zeros((n_individuals, n_components))
    # Get the coordinate of observed time points in the returned time grid.
    ti = []
    y = []
    for i, (id_, time_id) in enumerate(state_orders_test.items()):
        ti.append(np.array([np.argmin(np.abs(x - time_train))
                            for x in time_id]))
        y.append(tables_test[id_].T[ti[i] >= 0].values)
    y = np.concatenate(y).T.flatten()
    # SVD centralize if training was
    if v_centralized_train is not None:
        mean_hat = np.zeros((n_individuals,
                             n_features))
        for i, m in enumerate(tables_test.values()):
            mean_hat[i, :] = np.mean(m, axis=1)
        mean_hat_svd = mean_hat @ (v_centralized_train * v_centralized_train.T)
        for i, (id_, m) in enumerate(tables_test.items()):
            tables_test[id_] = (m.T - mean_hat_svd[[i]].flatten()).T
    # Project the each individual(s) data
    for s in range(n_components):
        for i, (id_, m) in enumerate(tables_test.items()):
            t_temp = ti[i] >= 0
            st_train = state_loading_train.values[np.array(ti[i])[t_temp], s]
            m_s = m.values[:, t_temp]
            id_projected[i, s] = fl_train.values[:, s] @ m_s @ st_train
            id_projected[i, s] = ((id_projected[i, s]
                                  / np.sum(st_train ** 2))
                                  / eigen_coeff_train[s])
            st_train = state_loading_train.iloc[ti[i][t_temp], [s]].values.T
            tbl_ = (fl_train.values[:, [s]] @ st_train)
            tables_test[id_] -= (eigen_coeff_train[s]
                                 * id_projected[i, s]
                                 * tbl_)
    id_projected = pd.DataFrame(id_projected,
                                tables_test.keys(),
                                state_loading_train.columns)
    return id_projected
