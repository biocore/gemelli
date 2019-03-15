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
from .utils import match
from .base import _BaseTransform
from deicode.preprocessing import rclr


class build(_BaseTransform):

    """
    This class can both build and RCLR
    transform tensors from 2D dataframes
    given count and mapping data. 
    
    A conditional measurement is given
    that identifies conditions measured
    multiple times over the same sample
    ID. Additionally a set of sample IDs
    must be provided. Any samples that are
    missing in a given condition are left
    as completely zero. The tensor is given
    in the shape (conditions, features,
    samples).

    The tensor is the RCLR transformed along
    each conditional slice. The output RCLR
    tensor is of the shape (samples, features,
    conditions).

    Parameters
    ----------
    pseudocount : float, optional
        A pseudocount used in the
        case that a sample, feature,
            and/or condition is completely
            missing from a vector slice in 
            the tensor formed.
    table : DataFrame
        table of non-negative count data
        rows = samples
        columns = features
    mapping : DataFrame
        mapping metadata for table
        rows = samples
        columns = metadata categories
    ID_col : str, int, or float
        category of sample IDs in metadata
    cond_col : str, int, or float
        category of conditional in metadata
    tensor : array-like, optional
        A premade tensor to RCLR transform.
        first dimension = conditions
        second dimension = features
        third dimension = samples

    Attributes
    -------
    ID_order : list
        order of IDs in tensor array
    feature_order : list
        order of features in tensor array
    cond_order : list
        order of conditions in tensor array
    tensor : array-like
        3rd order tensor of shape
        first dimension = conditions
        second dimension = features
        third dimension = samples
    TRCLR : array-like
        RCLR transformed 3rd order tensor
        of shape (transpose of input tensor).
        first dimension = samples
        second dimension = features
        third dimension = conditions

    Raises
    ------
    ValueError
        Raises an error if pseudocount
        is less than zero.
    ValueError
        if ID_col not in mapping cols
    ValueError
        if cond_col not in mapping cols
    ValueError
        Table is not 2-dimension
    ValueError
        Table contains negative values
    ValueError
        Table contains np.inf or -np.inf
    ValueError
        Table contains nans
    ValueError
        tensor is not 3-dimension
    ValueError
        tensor contains negative values
    ValueError
        tensor contains np.inf or -np.inf
    ValueError
        tensor contains nans
    Warning
        tensor contains no zeros
    Warning
        Table contains no zeros
    Warning
        If a conditional-sample pair
        has multiple IDs associated
        with it. In this case the
        default method is to sum them.
    Warning
        If total completely missing
        samples exceeds 50% of the
        data.

    References
    ----------
    .. [1] V. Pawlowsky-Glahn, J. J. Egozcue, R. Tolosana-Delgado (2015),
    Modeling and Analysis of Compositional Data, Wiley, Chichester, UK

    .. [2] C. Martino et al., A Novel Sparse Compositional Technique Reveals
    Microbial Perturbations. mSystems. 4 (2019), doi:10.1128/mSystems.00016-19.

    Examples
    --------

    To start with a 2D table.

    >>> t = Build()
    >>> t.fit(table,metadata,ID,condition)
    >>> t.TRCLR

    To RCLR transform with a
    prebuilt tensor.

    >>> T_counts = [[[ 0, 0, 0],
                        [ 0, 0, 0]],
                    [[10, 14, 43],
                    [ 41, 43, 14],
                    [[ 0 , 0 , 0 ],
                    [ 0,  0 , 0 ]]]
    >>> t = Build()
    >>> t.tensor
    >>> t.transform()
    >>> t.TRCLR

    """

    def __init__(self, pseudocount=1):

        """
        Parameters
        ----------
        pseudocount : float, optional
            A pseudocount used in the
            case that a sample, feature,
             and/or condition is completely
             missing from a vector slice in 
             the tensor formed.

        """
        if pseudocount < 0.0:
            raise ValueError("pseudocount should be larger than zero")

        self._pseudocount = pseudocount

    @property
    def pseudocount(self):

        """
        Pseudocount property
        allows pseudocount to
        be set explicitly.

        """

        return self._pseudocount

    @pseudocount.setter
    def pseudocount(self, value):

        """
        Set pseudocount value
        for property.

        Parameters
        ----------
        value : float
            pseudocount value

        Raises
        ------
        ValueError
            Raises an error if value is less than zero.

        Examples
        --------
        >>> t = Build()
        >>> t.pseudocount
        >>> t.pseudocount = 1

        """

        if value < 0.0:
            raise ValueError("pseudocount should be larger than zero")
        self._pseudocount = value

    @property
    def tensor(self):

        """
        pseudocount property
        allows pseudocount to
        be set explicitly.

        """

        return self._tensor

    @tensor.setter
    def tensor(self, tensor):

        """
        Set a tensor directly.

        Parameters
        ----------
        tensor : array-like
            first dimension = conditions
            second dimension = features
            third dimension = samples

        Raises
        ------
        ValueError
            Raises an error if value is less than zero.

        Examples
        --------
        >>> T_counts = [[[ 0, 0, 0],
                         [ 0, 0, 0]],
                       [[10, 14, 43],
                        [ 41, 43, 14],
                       [[ 0 , 0 , 0 ],
                        [ 0,  0 , 0 ]]]
        >>> t = Build()
        >>> t.tensor
        >>> t.tensor = T_counts

        """

        if len(tensor.shape) != 3:
            raise ValueError('tensor is not 3-dimension')

        if (tensor < 0).any():
            raise ValueError('tensor Contains Negative Values')

        if np.count_nonzero(np.isinf(tensor)) != 0:
            raise ValueError('tensor contains either np.inf or -np.inf')

        if np.count_nonzero(np.isnan(tensor)) != 0:
            raise ValueError('tensor contains nans')

        if np.count_nonzero(tensor) == 0:
            warnings.warn("tensor contains no zeros.", RuntimeWarning)
        
        self._tensor = tensor


    def fit(self,table,mapping,ID_col,cond_col):

        """
        This function transforms a 2D table
        into a 3rd-Order tensor in CLR space.

        Parameters
        ----------
        table : DataFrame
            table of non-negative count data
            rows = samples
            columns = features
        mapping : DataFrame
            mapping metadata for table
            rows = samples
            columns = metadata categories
        ID_col : str, int, or float
            category of sample IDs in metadata
        cond_col : str, int, or float
            category of conditional in metadata

        Returns
        -------
        self to abstract method in base

        Raises
        ------
        ValueError
            if ID_col not in mapping cols
        ValueError
            if cond_col not in mapping cols
        ValueError
            Table is not 2-dimensions
        ValueError
            Table contains negative values
        ValueError
            Table contains np.inf or -np.inf
        ValueError
            Table contains nans
        Warning
            Table contains no zeros

        Examples
        --------
        >>> t = Build()
        >>> t.fit(table,metadata,ID,condition)

        """

        if ID_col not in mapping.columns:
            raise ValueError("ID category not in metadata columns")

        if cond_col not in mapping.columns:
            raise ValueError("Conditional category not in metadata columns")

        if len(table.values.shape) != 2:
            raise ValueError('Table is not 2-dimensions')

        if (table.values < 0).any():
            raise ValueError('Table Contains Negative Values')

        if np.count_nonzero(np.isinf(table.values)) != 0:
            raise ValueError('Table contains either np.inf or -np.inf')

        if np.count_nonzero(np.isnan(table.values)) != 0:
            raise ValueError('Table contains nans')

        if np.count_nonzero(table.values) == 0:
            warnings.warn("Table contains no zeros.", RuntimeWarning)

        self.table = table.copy()
        self.mapping = mapping.copy()
        self.ID_col = ID_col
        self.cond_col = cond_col
        self._fit()

        return self

    def _fit(self):

        """
        This function forms a tensor
        with missing samples left as
        all zeros. It then passes that
        tensor into transform() for
        RCLR transformation.

        Raises
        ------
        Warning
            If a conditional-sample pair
            has multiple IDs associated
            with it. In this case the
            default method is to sum them.
        Warning
            If total completely missing
            samples exceeds 50% of the
            data.

        """

        # check that all indicies match & are unqiue 
        self.table,self.mapping = match(self.table,self.mapping)

        # order ids, cond, feats
        ID_order = sorted(set(self.mapping[self.ID_col]))
        cond_order = sorted(set(self.mapping[self.cond_col]))

        # empty tensor to fill 
        tensor = np.zeros((len(cond_order),
                           len(self.table.columns),
                           len(ID_order)))

        # fill tensor where possible
        table_index = np.array(self.table.index)
        table_array = self.table.values
        num_missing = 0 # check if fully missing samples
        for i,c_i in enumerate(cond_order):
            for j,ID_j in enumerate(ID_order):
                # get index ID assoc. in cond.
                idx = set(self.mapping[(self.mapping[self.ID_col].isin([ID_j])) \
                                & (self.mapping[self.cond_col].isin([c_i]))].index)
                if len(idx)>1:
                    warnings.warn('',join(["Condition ",str(c_i),
                                        " has multiple sample ",
                                        "with the same IDs ",
                                        str(ID_j)]), RuntimeWarning)
                elif len(idx)==0:
                    num_missing+=1
                    continue 
                # fill slice 
                tensor[i,:,j] = table_array[table_index == list(idx),:].sum(axis=0)
                
        # find percent totally missing samples
        self.perc_missing = num_missing/(len(cond_order)*len(ID_order))
        if self.perc_missing > 0.50:
            warnings.warn(''.join(["Total Missing Sample Exceeds 50% ",
                        "some conditions or samples may ",
                        "need to be removed."]), RuntimeWarning)

        # perform RCLR transformation
        self._tensor = tensor
        self.transform()

        # save intermediates
        self.ID_order  = ID_order
        self.feature_order  = self.table.columns
        self.cond_order  = cond_order

    def transform(self):

        """
        tensor wrapped for rclr transform
        will add pseudocount where samples
        are completely missing. The transform
        of those samples will be zero again.

        Raises
        ------
        ValueError
            tensor is not 3-dimensions
        ValueError
            tensor contains negative values
        ValueError
            tensor contains np.inf or -np.inf
        ValueError
            tensor contains nans
        Warning
            tensor contains no zeros

        References
        ----------
        .. [1] V. Pawlowsky-Glahn, J. J. Egozcue, R. Tolosana-Delgado (2015),
        Modeling and Analysis of Compositional Data, Wiley, Chichester, UK

        .. [2] C. Martino et al., A Novel Sparse Compositional Technique Reveals
        Microbial Perturbations. mSystems. 4 (2019), doi:10.1128/mSystems.00016-19.

        Examples
        --------

        To use directly with a
        prebuilt tensor.

        >>> T_counts = [[[ 0, 0, 0],
                         [ 0, 0, 0]],
                       [[10, 14, 43],
                        [ 41, 43, 14],
                       [[ 0 , 0 , 0 ],
                        [ 0,  0 , 0 ]]]
        >>> t = Build()
        >>> t.tensor = T_counts
        >>> t.transform()
        >>> t.TRCLR

        """

        if len(self._tensor.shape) != 3:
            raise ValueError('tensor is not 3-dimentional')

        if (self._tensor < 0).any():
            raise ValueError('tensor Contains Negative Values')

        if np.count_nonzero(np.isinf(self._tensor)) != 0:
            raise ValueError('tensor contains either np.inf or -np.inf')

        if np.count_nonzero(np.isnan(self._tensor)) != 0:
            raise ValueError('tensor contains nans')

        if np.count_nonzero(self._tensor) == 0:
            warnings.warn("tensor contains no zeros.", RuntimeWarning)
        
        # copy tensor to transform
        TRCLR = self._tensor.copy()

        # pseudocount totally missing samp: 
        # sum of all feat (time,samp)==0
        for i,j in np.argwhere(self._tensor.sum(axis=1) == 0):
            self._tensor[i,:,j] += self._pseudocount
        # add for any totally zero features (should not occur)
        if sum(self._tensor.sum(axis=0).sum(axis=1)==0) > 0:
            self._tensor[:,self._tensor.sum(axis=0).sum(axis=1)==0,:] += self._pseudocount
        # add for any totally zero timepoint (should not occur)
        if sum(self._tensor.sum(axis=2).sum(axis=1)==0) > 0:
            self._tensor[self._tensor.sum(axis=2).sum(axis=1)==0,:,:] += self._pseudocount

        # flatten
        TRCLR = np.concatenate([self._tensor[i,:,:].T 
                                for i in range(self._tensor.shape[0])],axis=0)

        # transform flat
        TRCLR = rclr().fit_transform(TRCLR)

        # re-shape tensor
        TRCLR = np.dstack([TRCLR[(i-1)*self._tensor.shape[-1]\
                                  :(i)*self._tensor.shape[-1]] 
                           for i in range(1,self._tensor.shape[0]+1)])

        # fill nan with zero
        TRCLR[np.isnan(TRCLR)] = 0 

        self.TRCLR = TRCLR
