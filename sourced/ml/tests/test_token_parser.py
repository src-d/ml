import pickle
import unittest

from sourced.ml.algorithms import TokenParser, NoopTokenParser


class TokenParserTests(unittest.TestCase):
    def setUp(self):
        self.tp = TokenParser(stem_threshold=4, max_token_length=20)

    def test_process_token(self):

        _max_token_length = self.tp.max_token_length
        self.tp.max_token_length = 100

        tokens = [
            ("sourced.ml.algorithms.uast_ids_to_bag",
             ["sourc", "sourcedml", "algorithm", "mlalgorithm",
              "uast", "ids", "idsto", "bag", "tobag"]),
            ("WORSTnameYOUcanIMAGINE", ['worst', 'name', 'you', 'can', 'imagin']),
            # Another bad example. Parser failed to parse it correctly
            ("SmallIdsToFoOo", ["small", "ids", 'idsto', 'fo', 'oo']),
            ("SmallIdFooo", ["small", "smallid", 'fooo', 'idfooo']),
            ("ONE_M0re_.__badId.example", ['one', 'onem', 're', 'bad', 'rebad',
                                           'badid', 'exampl', 'idexampl']),
            ("never_use_Such__varsableNames", ['never', 'use', 'such', 'varsabl', 'name']),
            ("a.b.c.d", ["a", "b", "c", "d"]),
            ("A.b.Cd.E", ['a', 'b', 'cd', 'e']),
            ("looong_sh_loooong_sh", ['looong', 'looongsh', 'loooong', 'shloooong', 'loooongsh']),
            ("sh_sh_sh_sh", ['sh', 'sh', 'sh', 'sh']),
            ("loooong_loooong_loooong", ['loooong', 'loooong', 'loooong'])
        ]

        for token, correct in tokens:
            res = list(self.tp.process_token(token))
            self.assertEqual(res, correct)
        self.tp.max_token_length = _max_token_length

    def test_split(self):
        self.assertEqual(list(self.tp.split("set for")), ["set", "for"])
        self.assertEqual(list(self.tp.split("set /for.")), ["set", "for"])
        self.assertEqual(list(self.tp.split("NeverHav")), ["never", "hav"])
        self.assertEqual(list(self.tp.split("PrintAll")), ["print", "all"])
        self.assertEqual(list(self.tp.split("PrintAllExcept")), ["print", "all", "except"])
        self.assertEqual(
            list(self.tp.split("print really long line")),
            # 'longli' is expected artifact due to edge effects
            ["print", "really", "long", "longli"])
        self.assertEqual(
            list(self.tp.split("set /for. *&PrintAll")),
            ["set", "for", "print", "all"])
        self.assertEqual(
            list(self.tp.split("JumpDown not Here")),
            ["jump", "down", "not", "here"])

    def test_stem(self):
        self.assertEqual(self.tp.stem("lol"), "lol")
        self.assertEqual(self.tp.stem("apple"), "appl")
        self.assertEqual(self.tp.stem("orange"), "orang")
        self.assertEqual(self.tp.stem("embedding"), "embed")
        self.assertEqual(self.tp.stem("Alfred"), "Alfred")
        self.assertEqual(self.tp.stem("Pluto"), "Pluto")

    def test_pickle(self):
        tp = pickle.loads(pickle.dumps(self.tp))
        self.assertEqual(tp.stem("embedding"), "embed")


class NoopTokenParserTests(unittest.TestCase):
    def setUp(self):
        self.tp = NoopTokenParser()

    def test_process_token(self):
        self.assertEqual(list(self.tp.process_token("abcdef")), ["abcdef"])
        self.assertEqual(list(self.tp.process_token("abcd_ef")), ["abcd_ef"])
        self.assertEqual(list(self.tp.process_token("abcDef")), ["abcDef"])


if __name__ == "__main__":
    unittest.main()
