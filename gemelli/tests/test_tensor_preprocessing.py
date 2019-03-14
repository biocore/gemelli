import unittest
import numpy as np
import numpy.testing as npt
from deicode.preprocessing import rclr
from gemelli.preprocessing import Build
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
        self._rclr = rclr()
        pass
    
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

        t = Build()
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
