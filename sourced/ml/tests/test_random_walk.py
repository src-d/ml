import unittest

import bblfsh

from sourced.ml.tests import models
from sourced.ml.algorithms.uast_struct_to_bag import Uast2RandomWalks
from sourced.ml.algorithms.uast_ids_to_bag import FakeVocabulary


class RandomWalkTests(unittest.TestCase):
    def setUp(self):
        self.bblfsh = bblfsh.BblfshClient("localhost:9432")
        self.uast = self.bblfsh.parse(models.SOURCE_PY).uast
        self.uast2walk = Uast2RandomWalks(p_explore_neighborhood=0.5,
                                          q_leave_neighborhood=0.5,
                                          n_walks=5,
                                          n_steps=19,
                                          node2index=FakeVocabulary(),
                                          seed=42)

    def test_rw(self):
        for walk in self.uast2walk(self.uast):
            for i in range(len(walk)-1):
                self.assertNotEqual(walk[i], walk[i+1],
                                    "Two neighbours nodes should not be the same")


if __name__ == "__main__":
    unittest.main()
