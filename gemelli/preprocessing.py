# ----------------------------------------------------------------------------
# Copyright (c) 2019--, gemelli development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np
import pandas as pd
from .utils import match
from .base import _BaseTransform
from deicode.preprocessing import rclr

@experimental(as_of="0.0.0")
class Build(_BaseTransform):

        """
        This class can both build and RCLR
        transform tensors from 2D dataframes
        given count and mapping data. 
        
        A conditional measurement is given
        that identfies conditions measured
        multiple times over the same sample
        ID. Additionally a set of sample IDs
        must be provided. Any samples that are
        missing in a given condition are left
        as completely zero. The tensor is given
        in the shape (conditions, features,
        samples).

        The tensor is the RCLR transformed along
        each conditional slice. The output RCLR
        tansor is of the shape (samples, features,
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
            first dimention = conditions
            second dimention = features
            third dimention = samples

        Returns
        -------
        ID_order : list
            order of IDs in tensor array
        feature_order : list
            order of features in tensor array
        cond_order : list
            order of conditions in tensor array
        Tensor : array-like
            3rd order tensor of shape
            first dimention = conditions
            second dimention = features
            third dimention = samples
        TRCLR : array-like
            RCLR transformed 3rd order tensor
            of shape (transpose of input tensor).
            first dimention = samples
            second dimention = features
            third dimention = conditions

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
            Table is not 2-dimentions
        ValueError
            Table contains negative values
        ValueError
            Table contains np.inf or -np.inf
        ValueError
            Table contains nans
        ValueError
            Tensor is not 3-dimentions
        ValueError
            Tensor contains negative values
        ValueError
            Tensor contains np.inf or -np.inf
        ValueError
            Tensor contains nans
        Warning
            Tensor contains no zeros
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
        >>> t.Tensor
        >>> t.tensor_rclr()
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

        self.pseudocount = pseudocount

    @property
    def pseudocount(self):

        """
        Pseudocount property
        allows pseudocount to
        be set explictly.

        """

        return self.pseudocount

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
        self.pseudocount = value

    @property
    def Tensor(self):

        """
        pseudocount property
        allows pseudocount to
        be set explictly.

        """

        return self.Tensor

    @Tensor.setter
    def Tensor(self, Tensor):

        """
        Set a tensor directly.

        Parameters
        ----------
        tensor : array-like
            first dimention = conditions
            second dimention = features
            third dimention = samples

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
        >>> t.Tensor
        >>> t.Tensor = T_counts

        """

        if len(Tensor.shape) != 3:
            raise ValueError('Tensor is not 3-dimentional')

        if (Tensor < 0).any():
            raise ValueError('Tensor Contains Negative Values')

        if np.count_nonzero(np.isinf(Tensor)) != 0:
            raise ValueError('Tensor contains either np.inf or -np.inf')

        if np.count_nonzero(np.isnan(Tensor)) != 0:
            raise ValueError('Tensor contains nans')

        if np.count_nonzero(Tensor) == 0:
            warnings.warn("Tensor contains no zeros.", RuntimeWarning)
        
        self.Tensor = Tensor


    def fit(self,table,mapping,ID_col,cond_col):

        """
        This function transforms a 2D table
        into a 3rd-Order Tensor in CLR space.

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
            Table is not 2-dimentions
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
            raise ValueError('Table is not 2-dimentions')

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
        tensor into tensor_rclr() for
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

        # empty Tensor to fill 
        Tensor = np.zeros((len(cond_order),
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
                Tensor[i,:,j] = table_array[table_index == list(idx),:].sum(axis=0)
                
        # find percent totally missing samples
        self.perc_missing = num_missing/(len(cond_order)*len(ID_order))
        if self.perc_missing > 0.50:
            warnings.warn(''.join(["Total Missing Sample Exceeds 50% ",
                        "some conditions or samples may ",
                        "need to be removed."]), RuntimeWarning)

        # perform RCLR transformation
        self.Tensor = Tensor
        self.tensor_rclr()

        # save intermediates
        self.ID_order  = ID_order
        self.feature_order  = self.table.columns
        self.cond_order  = cond_order

    def tensor_rclr(self):

        """
        Tensor wrapped for rclr transform
        will add pseudocount where samples
        are completely missing. The transform
        of those samples will be zero again.

        Raises
        ------
        ValueError
            Tensor is not 3-dimentions
        ValueError
            Tensor contains negative values
        ValueError
            Tensor contains np.inf or -np.inf
        ValueError
            Tensor contains nans
        Warning
            Tensor contains no zeros

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
        >>> t.Tensor
        >>> t.tensor_rclr()
        >>> t.TRCLR

        """

        if len(self.Tensor.shape) != 3:
            raise ValueError('Tensor is not 3-dimentional')

        if (self.Tensor < 0).any():
            raise ValueError('Tensor Contains Negative Values')

        if np.count_nonzero(np.isinf(self.Tensor)) != 0:
            raise ValueError('Tensor contains either np.inf or -np.inf')

        if np.count_nonzero(np.isnan(self.Tensor)) != 0:
            raise ValueError('Tensor contains nans')

        if np.count_nonzero(self.Tensor) == 0:
            warnings.warn("Tensor contains no zeros.", RuntimeWarning)
        
        # copy tensor to transform
        TRCLR = self.Tensor.copy()

        # pseudocount totally missing samp: 
        # sum of all feat (time,samp)==0
        if self.perc_missing > 0.0:
            for i,j in np.argwhere(self.Tensor.sum(axis=1) == 0):
                self.Tensor[i,:,j]+=self.pseudocount
        # add for any totally zero features (should not occur)
        if sum(self.Tensor.sum(axis=0).sum(axis=1)==0) > 0:
            self.Tensor[:,self.Tensor.sum(axis=0).sum(axis=1)==0,:] += self.pseudocount
        # add for any totally zero timepoint (should not occur)
        if sum(self.Tensor.sum(axis=2).sum(axis=1)==0) > 0:
            self.Tensor[self.Tensor.sum(axis=2).sum(axis=1)==0,:,:] += self.pseudocount

        # flatten
        TRCLR = np.concatenate([self.Tensor[i,:,:].T 
                                for i in range(self.Tensor.shape[0])],axis=0)

        # transform flat
        TRCLR = rclr().fit_transform(TRCLR)

        # re-shape tensor
        TRCLR = np.dstack([TRCLR[(i-1)*self.Tensor.shape[-1]\
                                  :(i)*self.Tensor.shape[-1]] 
                           for i in range(1,self.Tensor.shape[0]+1)])

        # fill nan with zero
        TRCLR[np.isnan(TRCLR)] = 0 

        self.TRCLR = TRCLR
