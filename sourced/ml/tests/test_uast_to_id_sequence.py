import unittest

from bblfsh import BblfshClient

from sourced.ml.algorithms import Uast2IdSequence, NoopTokenParser
from sourced.ml.tests.models import SOURCE_PY


class Uast2IdSequenceTest(unittest.TestCase):
    def setUp(self):
        self.uast2id_sequence = Uast2IdSequence(token_parser=NoopTokenParser())
        self.uast = BblfshClient("0.0.0.0:9432").parse(SOURCE_PY).uast

    def test_result(self):
        correct = ['sys', 'setup_logging', 'modelforge.logs', 'utmain', 'modules', 'sys',
                   '__package__', 'utmain',  '__spec__', 'utmain',  'namedtuple', 'collections',
                   'ModuleSpec', 'namedtuple', '__spec__', 'utmain', 'ModuleSpec', 'ModuleSpec',
                   'utmain', 'setup', 'setup_logging']
        res = self.uast2id_sequence(self.uast)
        self.assertEqual(res, self.uast2id_sequence.concat(correct))


if __name__ == '__main__':
    unittest.main()
