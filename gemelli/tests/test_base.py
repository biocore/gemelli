import unittest
from gemelli.base import _BaseTransform, _BaseImpute


class Test_BaseTransform(unittest.TestCase):
    def test_no_instantiation(self):
        class Foo(_BaseTransform):
            pass


class Test_BaseImpute(unittest.TestCase):
    def test_no_instantiation(self):
        class Foo_boo(_BaseImpute):
            pass
