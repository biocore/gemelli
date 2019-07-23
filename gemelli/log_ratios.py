import numpy as np
np.seterr(divide='ignore')


def log_ratio(numerator, denominator, pseudocount=None):
    """
    Take the log-ratio of the numerator and the denominator.

    Parameters
    ----------
    numerator : array_like, float
    denominator : array_like, float
    pseudocount : float, optional

    Returns
    ----------
    log-ratio: array_like, float
        log ratio of numerator and denominator

    Raises
    ----------
    ValueError
        If shape numerator and denominator are not equal.
    ValueError
        If there is a pseudocount and the log of the numerator or
        denominator contains non-finite values

    Examples
    ----------
    TODO

    """

    numerator = np.atleast_1d(numerator)
    denominator = np.atleast_1d(denominator)

    if numerator.shape != denominator.shape:
        raise ValueError('The numerator and denominator must be of',
                         ' equal length.')

    if pseudocount is not None:
        numerator = np.log(numerator + pseudocount)
        denominator = np.log(denominator + pseudocount)
        # ensure log-ratio has no zeros
        # this only a problem in
        if not all(np.isfinite(numerator)):
            raise ValueError('Log of numerator is non-finite')
        if not all(np.isfinite(denominator)):
            raise ValueError('Log of denominator is non-finite')
    else:
        numerator = np.log(numerator)
        denominator = np.log(denominator)

    return numerator - denominator


def percentile_ratio(mat, ranks, pseudocount=None,
                     percent=25, interpolation='midpoint'):
    """
    Sum samples of count matrix with the upper and lower percentile of
    ranked features and then perform a log-ratio between them.

    Parameters
    ----------
    mat : array_like, float
       a matrix of counts where
       rows = features and
       columns = samples
    ranks : array_like, float
        a matrix of feature ranks
    pseudocount: float, optional
        Add a pseudocount before taking the log-ratio. Default is None.
    percent: int, optional [1,100]
        The upper and lower percentile of ranks for the log-ratio.
        The default is quartiles (i.e. 25).
    interpolation:
        Interpolation method to use when the desired percentile lies
        between two data points. Default is 'midpoint'

    Returns
    ----------
    log_ratio: array_like, float
        The log-ratio between the upper and lower percentiales
        of ranked features summed across all samples. Where the shape is
        equal to the number of columns of mat.

    Raises
    ----------
    ValueError
        If the number of ranks and features are not equal.
    ValueError
        If percent is less than or equal to zero or greater than 50


    Examples
    ----------
    TODO

    """

    mat = np.atleast_2d(mat)
    ranks = np.atleast_1d(ranks)

    if len(ranks) != mat.shape[0]:
        raise ValueError('The number of ranks and features are not equal.')
    if percent <= 0 or percent > 50:
        raise ValueError('Percent must be greater than zero ',
                         'and less than 50.')
    # calculate the percentiles
    perc = np.percentile(ranks, range(0, 100, percent),
                         interpolation=interpolation)
    # sum across samples for upper percentile
    numerator = mat[ranks >= perc[-1]].sum(axis=0)
    # sum across samples for lower percentile
    denominator = mat[ranks <= perc[1]].sum(axis=0)
    # compute log-ratio
    return log_ratio(numerator, denominator, pseudocount=pseudocount)
