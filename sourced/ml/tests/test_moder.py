import unittest

import bblfsh

from sourced.ml.transformers import Moder
from sourced.ml.tests.models import MODER_FUNC


class ModerTest(unittest.TestCase):
    def test_extract_functions_from_uast(self):
        client = bblfsh.BblfshClient("localhost:9432")
        uast = client.parse(MODER_FUNC).uast
        functions = list(Moder(mode="func").extract_functions_from_uast(uast))
        self.assertEqual(len(functions), 3)

        function_names = ["func_a", "func_b", "func_c"]
        for f in functions:
            self.assertTrue(f[0].token in function_names)


if __name__ == "__main__":
    unittest.main()
