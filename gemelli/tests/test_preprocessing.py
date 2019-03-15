import unittest
import numpy as np
import pandas as pd
import numpy.testing as npt
from deicode.preprocessing import rclr
from gemelli.preprocessing import build
from skbio.stats.composition import closure, clr


class Testpreprocessing(unittest.TestCase):

    def setUp(self):

        self.cdata1 = np.array([[2, 2, 6],
                                [4, 4, 2]])
        
        self.cdata2 = [[3, 3, 0], [0, 4, 2]]
        
        self.true2 = np.array([[0.0, 0.0, np.nan],
                               [np.nan, 0.34657359, 
                                -0.34657359]])
        
        self.T_true2 = [[[ 0., 0., 0.],[ 0., 0., 0.]],
                        [[-0.14384104, -0.14384104, -0.14384104],
                         [ 0.14384104, 0.14384104, 0.14384104]],
                        [[ 0. , 0. , 0. ],[ 0.,  0. , 0. ]]]

        self.bad1 = np.array([1, 2, -1])

        self.tenres = np.array([[[1, 2, 3],
                                [4, 5, 6]],
                            [[7, 8, 9],
                                [10, 11, 12]],
                            [[13 , 14 , 15],
                                [16,  17, 18]]])

        self.clrres = np.array([[[-1.76959918],
                                [-1.076452  ],
                                [-0.67098689],
                                [ 0.17631097],
                                [ 0.30984236],
                                [ 0.4276254 ],
                                [ 0.79535018],
                                [ 0.86945815],
                                [ 0.93845102]],
                            [[-0.88804481],
                                [-0.66490126],
                                [-0.48257971],
                                [ 0.02824592],
                                [ 0.1235561 ],
                                [ 0.21056747],
                                [ 0.49824955],
                                [ 0.55887417],
                                [ 0.61603258]]])
        self.t = build()
        self._rclr = rclr()
        pass

    def test_build(self):

        shape_ = self.tenres.shape[0]
        M_counts = np.concatenate([self.tenres[i,:,:].T 
                                for i in range(shape_)],
                                axis=0).T
        mapping = np.array([[0,0,1,1,2,2],
                            [0,1,2,0,1,2]])
        mapping =pd.DataFrame(mapping.T,
                            columns=['Cond','ID'])
        table = pd.DataFrame(M_counts)
        
        self.t.fit(table,mapping,'ID','Cond')
        npt.assert_allclose(self.t.tensor.astype(float).reshape(1, 9, 2),
                            M_counts.T.astype(float).reshape(1, 9, 2))
        self.t.transform()
        npt.assert_allclose(np.around(self.t.TRCLR.astype(float),3),
                            np.around(self.clrres.astype(float),3))

    
    def test_matrix_rclr(self):

        # test clr works the same if there are no zeros
        cmat = self._rclr.fit_transform(self.cdata1)
        npt.assert_allclose(cmat, clr(self.cdata1.copy()))

        # test a case with zeros
        cmat = self._rclr.fit_transform(self.cdata2)
        npt.assert_allclose(cmat, self.true2)

        with self.assertRaises(ValueError):
            self._rclr.fit_transform(self.bad1)
   
    def test_transform(self):

        t = build()
        # test flat clr works the same if there are no zeros
        t.tensor = np.stack([self.cdata1 for i in range(3)])
        t.transform()
        T_cmat = t.TRCLR
        test_T = np.stack([self.cdata1 for i in range(3)])
        clr_test_T = clr(np.concatenate([test_T[i,:,:].T 
                                          for i in range(test_T.shape[0])],axis=0))
        clr_test_T =  np.dstack([clr_test_T[(i-1)*test_T.shape[-1]:(i)*test_T.shape[-1]] 
                                  for i in range(1,test_T.shape[0]+1)])
        npt.assert_allclose(T_cmat, clr_test_T)
        
        # test a case with zeros
        t.tensor = np.stack([self.cdata2 for i in range(3)])
        t.transform()
        T_cmat = t.TRCLR 
        npt.assert_allclose(T_cmat, self.T_true2)

        with self.assertRaises(ValueError):
            t.tensor = np.stack([self.bad1 for i in range(3)])
            t.transform()
