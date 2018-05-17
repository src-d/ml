import random
from collections import defaultdict

from sourced.ml.algorithms.uast_ids_to_bag import FakeVocabulary, Uast2BagBase
from sourced.ml.algorithms import uast2sequence


class Uast2StructBagBase(Uast2BagBase):
    SEP = ">"

    def __init__(self, stride, seq_len, node2index=None):
        self._node2index = node2index if node2index is not None else FakeVocabulary()
        self._stride = stride
        if not isinstance(seq_len, (int, tuple, list)):
            raise TypeError("Unexpected type of seq_len: %s" % type(seq_len))
        self._seq_lens = [seq_len] if isinstance(seq_len, int) else seq_len

    @property
    def node2index(self):
        return self._node2index


class Node2InternalType:
    # FIXME(zurk): change to simple function. Vadim Markovtsev comments:
    # > would rather made this a simple function and change roles2index
    # type from [] to callable. Saves time to understand.
    def __getitem__(self, item):
        return item.internal_type


class UastSeq2Bag(Uast2StructBagBase):
    """
    DFS traversal + preserves the order of node children.
    """

    def __init__(self, stride=1, seq_len=(3, 4), node2index=None):
        _node2index = Node2InternalType() if node2index is None else node2index
        super().__init__(stride, seq_len, _node2index)

    def __call__(self, uast):
        bag = defaultdict(int)
        node_sequence = uast2sequence(uast)

        # convert to str - requirement from wmhash.BagsExtractor
        node_sequence = [self.node2index[n] for n in node_sequence]

        for seq_len in self._seq_lens:
            for i in range(0, len(node_sequence) - seq_len + 1, self._stride):
                key = self.SEP.join(node_sequence[i:i + seq_len])
                bag[key] += 1
        return bag


class Node:
    def __init__(self, parent=None, internal_type=None):
        self.parent = parent
        self.internal_type = internal_type
        self.children = []

    @property
    def neighbours(self):
        neighbours = []
        if self.parent is not None:
            neighbours.append(self.parent)
        neighbours.extend(self.children)
        return neighbours


class Uast2RandomWalks:
    """
    Generation of random walks for UAST.
    """

    def __init__(self, p_explore_neighborhood, q_leave_neighborhood, n_walks, n_steps,
                 node2index=None, seed=None):
        """
        Related article: https://arxiv.org/abs/1607.00653

        :param p_explore_neighborhood: return parameter, p. Parameter p controls the likelihood of\
                                       immediately revisiting a node in the walk. Setting it to a\
                                       high value (> max(q, 1)) ensures that we are less likely to\
                                       sample an already visited node in the following two steps\
                                       (unless the next node in the walk had no other neighbor).\
                                       This strategy encourages moderate exploration and avoids\
                                       2-hop redundancy in sampling.
        :param q_leave_neighborhood: in-out parameter, q. Parameter q allows the search to\
                                     differentiate between “inward” and “outward” nodes. Such \
                                     walks obtain a local view of the underlying graph with \
                                     respect to the start node in the walk and approximate BFS \
                                     behavior in the sense that our samples comprise of nodes \
                                     within a small locality.
        :param n_walks: Number of walks from each node.
        :param n_steps: Number of steps in walk.
        """
        self.p_explore_neighborhood = p_explore_neighborhood
        self.q_leave_neighborhood = q_leave_neighborhood
        self.n_walks = n_walks
        self.n_steps = n_steps
        self.node2index = node2index if node2index is not None else Node2InternalType()
        if seed is not None:
            random.seed(seed)

    def __call__(self, uast):
        starting_nodes = self.prepare_starting_nodes(uast)
        for i in range(self.n_walks):
            for start_node in starting_nodes:
                yield self.random_walk(start_node)

    @staticmethod
    def _extract_node(node, parent):
        return Node(parent=parent, internal_type=node.internal_type)

    def prepare_starting_nodes(self, uast):
        starting_nodes = []
        root = self._extract_node(uast, None)
        stack = [(root, uast)]
        while stack:
            parent, parent_uast = stack.pop()
            children_nodes = [self._extract_node(child, parent) for child in parent_uast.children]
            parent.children = children_nodes
            stack.extend(zip(children_nodes, parent_uast.children))
            starting_nodes.append(parent)

        return starting_nodes

    def random_walk(self, node):
        walk = [node]
        while len(walk) < self.n_steps:
            walk.append(self.alias_sample(walk))

        walk = [self.node2index[n] for n in walk]
        return walk

    def alias_sample(self, walk):
        """
        Compare to node2vec this sampling is a bit simpler because there is no loop in tree ->
        so there are only 2 options with unnormalized probabilities 1/p & 1/q
        Related article: https://arxiv.org/abs/1607.00653

        :param walk: list of visited nodes
        :return: next node to visit
        """
        last_node = walk[-1]  # correspond to node v in article

        if len(walk) == 1:
            choice_list = last_node.children
            if last_node.parent is not None:
                choice_list.append(last_node.parent)
            if len(choice_list) == 0:
                return last_node
            return random.choice(last_node.children)

        threshold = (1 / self.p_explore_neighborhood)
        threshold /= (threshold + len(last_node.children) / self.q_leave_neighborhood)

        if random.random() <= threshold:
            # With threshold probability we need to return back to previous node.
            return walk[-2]  # Node from previous step. Correspond to node t in article.

        return random.choice(last_node.neighbours)


class UastRandomWalk2Bag(Uast2StructBagBase):
    def __init__(self, p_explore_neighborhood=0.79, q_leave_neighborhood=0.82, n_walks=2,
                 n_steps=10, stride=1, seq_len=(2, 3), seed=42):
        super().__init__(stride, seq_len)
        self.uast2walks = Uast2RandomWalks(p_explore_neighborhood=p_explore_neighborhood,
                                           q_leave_neighborhood=q_leave_neighborhood,
                                           n_walks=n_walks, n_steps=n_steps, seed=seed)

    def __call__(self, uast):
        bag = defaultdict(int)
        for walk in self.uast2walks(uast):
            for seq_len in self._seq_lens:
                for i in range(0, len(walk) - seq_len + 1, self._stride):
                    # convert to str - requirement from wmhash.BagsExtractor
                    bag[self.SEP.join(walk[i:i + seq_len])] += 1
        return bag
