import unittest

from bblfsh import BblfshClient

from sourced.ml.algorithms import Uast2IdTreeDistance, Uast2IdLineDistance, NoopTokenParser
from sourced.ml.tests.models import SOURCE_PY


class Uast2IdTreeDistanceTest(unittest.TestCase):
    def setUp(self):
        self.uast2role_id_pairs = Uast2IdTreeDistance(token_parser=NoopTokenParser(),
                                                      max_distance=4)
        self.uast = BblfshClient("0.0.0.0:9432").parse(SOURCE_PY).uast

    def test_result(self):
        correct = [(('__spec__', 'ModuleSpec'), 2),
                   (('__spec__', 'ModuleSpec'), 2),
                   (('__spec__', 'ModuleSpec'), 3),
                   (('__spec__', '__package__'), 2),
                   (('collections', 'ModuleSpec'), 2),
                   (('collections', 'ModuleSpec'), 2),
                   (('collections', 'ModuleSpec'), 3),
                   (('collections', '__spec__'), 2),
                   (('modules', 'modelforge.logs'), 3),
                   (('namedtuple', 'ModuleSpec'), 3),
                   (('namedtuple', 'ModuleSpec'), 3),
                   (('namedtuple', 'ModuleSpec'), 3),
                   (('namedtuple', 'ModuleSpec'), 3),
                   (('namedtuple', '__spec__'), 3),
                   (('namedtuple', '__spec__'), 3),
                   (('namedtuple', 'collections'), 3),
                   (('namedtuple', 'collections'), 3),
                   (('setup', 'modelforge.logs'), 3),
                   (('setup_logging', 'modelforge.logs'), 3),
                   (('sys', 'modelforge.logs'), 3),
                   (('sys', 'modules'), 3),
                   (('utmain', 'ModuleSpec'), 2),
                   (('utmain', 'ModuleSpec'), 3),
                   (('utmain', 'ModuleSpec'), 3),
                   (('utmain', '__package__'), 3),
                   (('utmain', '__package__'), 3),
                   (('utmain', '__spec__'), 3),
                   (('utmain', '__spec__'), 3),
                   (('utmain', '__spec__'), 3),
                   (('utmain', 'collections'), 3),
                   (('utmain', 'modelforge.logs'), 2),
                   (('utmain', 'modelforge.logs'), 2),
                   (('utmain', 'modules'), 3),
                   (('utmain', 'modules'), 3),
                   (('utmain', 'setup'), 3),
                   (('utmain', 'setup'), 3),
                   (('utmain', 'setup_logging'), 3),
                   (('utmain', 'setup_logging'), 3),
                   (('utmain', 'sys'), 3),
                   (('utmain', 'sys'), 3)]

        res = sorted(self.uast2role_id_pairs(self.uast))
        self.assertEqual(res, correct)


class Uast2IdLineDistanceTest(unittest.TestCase):
    def setUp(self):
        self.uast2role_id_pairs = Uast2IdLineDistance(token_parser=NoopTokenParser(),
                                                      max_distance=3)
        self.uast = BblfshClient("0.0.0.0:9432").parse(SOURCE_PY).uast

    def test_result(self):
        correct = [(('__package__', 'ModuleSpec'), 2),
                   (('__spec__', 'ModuleSpec'), 0),
                   (('__spec__', 'ModuleSpec'), 1),
                   (('__spec__', 'ModuleSpec'), 1),
                   (('__spec__', 'ModuleSpec'), 2),
                   (('__spec__', '__package__'), 0),
                   (('collections', 'ModuleSpec'), 1),
                   (('collections', 'ModuleSpec'), 2),
                   (('collections', '__package__'), 1),
                   (('collections', '__spec__'), 1),
                   (('collections', '__spec__'), 2),
                   (('modules', '__package__'), 1),
                   (('modules', '__spec__'), 1),
                   (('modules', 'collections'), 2),
                   (('namedtuple', 'ModuleSpec'), 0),
                   (('namedtuple', 'ModuleSpec'), 1),
                   (('namedtuple', 'ModuleSpec'), 1),
                   (('namedtuple', 'ModuleSpec'), 2),
                   (('namedtuple', 'ModuleSpec'), 2),
                   (('namedtuple', '__package__'), 1),
                   (('namedtuple', '__package__'), 2),
                   (('namedtuple', '__spec__'), 1),
                   (('namedtuple', '__spec__'), 1),
                   (('namedtuple', '__spec__'), 2),
                   (('namedtuple', '__spec__'), 2),
                   (('namedtuple', 'collections'), 0),
                   (('namedtuple', 'collections'), 1),
                   (('namedtuple', 'modules'), 2),
                   (('setup_logging', 'modelforge.logs'), 0),
                   (('setup_logging', 'setup'), 1),
                   (('sys', '__package__'), 1),
                   (('sys', '__spec__'), 1),
                   (('sys', 'collections'), 2),
                   (('sys', 'modelforge.logs'), 2),
                   (('sys', 'modules'), 0),
                   (('sys', 'namedtuple'), 2),
                   (('sys', 'setup_logging'), 2),
                   (('utmain', 'ModuleSpec'), 0),
                   (('utmain', 'ModuleSpec'), 1),
                   (('utmain', 'ModuleSpec'), 1),
                   (('utmain', 'ModuleSpec'), 1),
                   (('utmain', 'ModuleSpec'), 2),
                   (('utmain', 'ModuleSpec'), 2),
                   (('utmain', 'ModuleSpec'), 2),
                   (('utmain', '__package__'), 0),
                   (('utmain', '__package__'), 0),
                   (('utmain', '__package__'), 1),
                   (('utmain', '__spec__'), 0),
                   (('utmain', '__spec__'), 0),
                   (('utmain', '__spec__'), 0),
                   (('utmain', '__spec__'), 1),
                   (('utmain', '__spec__'), 2),
                   (('utmain', 'collections'), 1),
                   (('utmain', 'collections'), 1),
                   (('utmain', 'collections'), 2),
                   (('utmain', 'collections'), 2),
                   (('utmain', 'modules'), 0),
                   (('utmain', 'modules'), 1),
                   (('utmain', 'modules'), 1),
                   (('utmain', 'namedtuple'), 1),
                   (('utmain', 'namedtuple'), 1),
                   (('utmain', 'namedtuple'), 1),
                   (('utmain', 'namedtuple'), 2),
                   (('utmain', 'namedtuple'), 2),
                   (('utmain', 'namedtuple'), 2),
                   (('utmain', 'namedtuple'), 2),
                   (('utmain', 'sys'), 0),
                   (('utmain', 'sys'), 1),
                   (('utmain', 'sys'), 1)]

        res = sorted(self.uast2role_id_pairs(self.uast))
        self.assertEqual(res, correct)


if __name__ == '__main__':
    unittest.main()
