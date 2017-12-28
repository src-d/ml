import unittest

from sourced.ml.transformers import Transformer


class DumpTransformer(Transformer):
    call_ids = []

    def __init__(self, id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global transformer_id
        self.id = id

    def __call__(self, *args, **kwargs):
        DumpTransformer.call_ids.append(self.id)
        return self.id


class TransformerTest(unittest.TestCase):

    def setUp(self):
        self.transformers_number = 8
        self.transformers_linear = [DumpTransformer(i) for i in range(self.transformers_number)]
        self.pipeline_linear = self.transformers_linear[0]
        for transformer in self.transformers_linear[1:]:
            self.pipeline_linear = self.pipeline_linear.link(transformer)
        """
        Tree structure:
        0 - 2 --- 5 - 6
         \   \ \
          1   3 4
        """
        self.transformers_tree = [DumpTransformer(i) for i in range(7)]
        self.pipeline_tree = self.transformers_tree[0] \
            .link(self.transformers_tree[2]) \
            .link(self.transformers_tree[5]) \
            .link(self.transformers_tree[6])

        self.transformers_tree[0].link(self.transformers_tree[1])
        self.transformers_tree[2].link(self.transformers_tree[3])
        self.transformers_tree[2].link(self.transformers_tree[4])

    def test_explain_pipeline(self):
        t = Transformer(False)
        self.assertFalse(t.explained)
        t2 = Transformer(True)
        self.assertTrue(t2.explained)
        p = t2.link(t)
        self.assertFalse(p.explained)

    def test_children(self):
        t = Transformer()
        t_c1 = Transformer()
        t_c2 = Transformer()
        t.link(t_c1)
        t.link(t_c2)
        self.assertEqual(t.children, (t_c1, t_c2))
        self.assertEqual(t_c1.children, tuple())
        self.assertEqual(t_c2.children, tuple())

    def test_parent(self):
        t = Transformer(False)
        t2 = Transformer(True)
        t2.link(t)
        self.assertEqual(t.parent, t2)
        self.assertIsNone(t2.parent)

    def test_unlink(self):
        t = Transformer(False)
        t2 = Transformer(True)
        t2.link(t)
        self.assertEqual(t.parent, t2)
        self.assertEqual(t2.children, (t,))
        t2.unlink(t)
        self.assertIsNone(t.parent)
        self.assertEqual(t2.children, tuple())

    def test_path(self):
        self.assertEqual(self.pipeline_linear.path(), self.transformers_linear)
        self.assertEqual([t.id for t in self.pipeline_tree.path()], [0, 2, 5, 6])
        self.assertEqual([t.id for t in self.transformers_tree[6].path()], [0, 2, 5, 6])
        self.assertEqual([t.id for t in self.transformers_tree[4].path()], [0, 2, 4])

    def test_explode(self):
        DumpTransformer.call_ids = []
        self.pipeline_linear.explode()
        self.assertEqual(DumpTransformer.call_ids, list(range(self.transformers_number)))
        for t in self.transformers_linear:
            DumpTransformer.call_ids = []
            t.explode()
            self.assertEqual(DumpTransformer.call_ids, list(range(self.transformers_number)))

        expected = {0: {0, 2, 5, 6, 3, 4, 1},
                    1: {0, 1},
                    2: {0, 2, 5, 6, 3, 4},
                    3: {0, 2, 3},
                    4: {0, 2, 4},
                    5: {0, 2, 5, 6},
                    6: {0, 2, 5, 6}}
        for t in self.transformers_tree:
            DumpTransformer.call_ids = []
            t.explode()
            self.assertEqual(expected[t.id], set(DumpTransformer.call_ids))

    def test_execute(self):
        DumpTransformer.call_ids = []
        self.pipeline_linear.execute()
        self.assertEqual(DumpTransformer.call_ids, list(range(self.transformers_number)))
        for t in self.transformers_linear:
            DumpTransformer.call_ids = []
            t.execute()
            self.assertEqual(DumpTransformer.call_ids, list(range(t.id+1)))

        expected = {0: {0},
                    1: {0, 1},
                    2: {0, 2},
                    3: {0, 2, 3},
                    4: {0, 2, 4},
                    5: {0, 2, 5},
                    6: {0, 2, 5, 6}}
        for t in self.transformers_tree:
            DumpTransformer.call_ids = []
            t.execute()
            self.assertEqual(expected[t.id], set(DumpTransformer.call_ids))

    def test_graph(self):
        import io
        out = io.StringIO()
        self.transformers_linear[0].graph(stream=out)
        graph = out.getvalue()
        self.assertEqual(graph,
                         'digraph source-d {\n'
                         '	"DumpTransformer 1" -> "DumpTransformer 2"\n'
                         '	"DumpTransformer 2" -> "DumpTransformer 3"\n'
                         '	"DumpTransformer 3" -> "DumpTransformer 4"\n'
                         '	"DumpTransformer 4" -> "DumpTransformer 5"\n'
                         '	"DumpTransformer 5" -> "DumpTransformer 6"\n'
                         '	"DumpTransformer 6" -> "DumpTransformer 7"\n'
                         '	"DumpTransformer 7" -> "DumpTransformer 8"\n'
                         '}\n')
        out = io.StringIO()
        self.transformers_tree[0].graph(stream=out)
        tree = out.getvalue()
        self.assertEqual(tree,
                         'digraph source-d {\n'
                         '	"DumpTransformer 1" -> "DumpTransformer 2"\n'
                         '	"DumpTransformer 1" -> "DumpTransformer 3"\n'
                         '	"DumpTransformer 2" -> "DumpTransformer 4"\n'
                         '	"DumpTransformer 2" -> "DumpTransformer 5"\n'
                         '	"DumpTransformer 2" -> "DumpTransformer 6"\n'
                         '	"DumpTransformer 4" -> "DumpTransformer 7"\n'
                         '}\n')


if __name__ == "__main__":
    unittest.main()
