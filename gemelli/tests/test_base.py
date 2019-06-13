import unittest
from gemelli.base import _BaseConstruct, _BaseImpute


class Test_BaseConstruct(unittest.TestCase):
    def test_no_instantiation(self):
        class Foo(_BaseConstruct):
            pass


class Test_BaseImpute(unittest.TestCase):
    def test_no_instantiation(self):
        class Foo_boo(_BaseImpute):
            pass
