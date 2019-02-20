import unittest
import numpy as np
import pandas as pd
from gemelli.utils import match
import pandas.util.testing as pdt

class TestMatch(unittest.TestCase):

    def test_match(self):
        """ Match on dense pandas tables,
        taken from gneiss (now dep.)
        https://github.com/biocore/
        gneiss/blob/master/gneiss/
        tests/test_util.py
         """
        table = pd.DataFrame([[0, 0, 1, 1],
                              [2, 2, 4, 4],
                              [5, 5, 3, 3],
                              [0, 0, 0, 1]],
                             index=['s1', 's2', 's3', 's4'],
                             columns=['o1', 'o2', 'o3', 'o4'])
        metadata = pd.DataFrame([['a', 'control'],
                                 ['b', 'control'],
                                 ['c', 'diseased'],
                                 ['d', 'diseased']],
                                index=['s1', 's2', 's3', 's4'],
                                columns=['Barcode', 'Treatment'])
        exp_table, exp_metadata = table, metadata
        res_table, res_metadata = match(table, metadata)

        # make sure that the metadata and table indeces match
        pdt.assert_index_equal(res_table.index, res_metadata.index)

        res_table = res_table.sort_index()
        exp_table = exp_table.sort_index()

        res_metadata = res_metadata.sort_index()
        exp_metadata = exp_metadata.sort_index()

        pdt.assert_frame_equal(exp_table, res_table)
        pdt.assert_frame_equal(exp_metadata, res_metadata)

    def test_match_empty(self):
        table = pd.DataFrame([[0, 0, 1, 1],
                              [2, 2, 4, 4],
                              [5, 5, 3, 3],
                              [0, 0, 0, 1]],
                             index=['s1', 's2', 's3', 's4'],
                             columns=['o1', 'o2', 'o3', 'o4'])
        metadata = pd.DataFrame([['a', 'control'],
                                 ['b', 'control'],
                                 ['c', 'diseased'],
                                 ['d', 'diseased']],
                                index=['a1', 'a2', 'a3', 'a4'],
                                columns=['Barcode', 'Treatment'])

        with self.assertRaises(ValueError):
            match(table, metadata)


