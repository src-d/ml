from typing import Iterable, Tuple

import bblfsh

from sourced.ml.algorithms.uast_ids_to_bag import UastIds2Bag
from sourced.ml.utils import bblfsh_roles


class Uast2RoleIdPairs(UastIds2Bag):
    """
    Converts a UAST to a list of pairs. Pair is identifier and role, where role is Node role
    where identifier was found.

    __call__ is overridden here and returns list instead of bag-of-words (dist).
    """

    def __init__(self, token2index=None, token_parser=None):
        """
        :param token2index: The mapping from tokens to token key. If None, no mapping is performed.
        :param token_parser: Specify token parser if you want to use a custom one. \
            :class:'TokenParser' is used if it is not specified.

        """
        super().__init__(token2index=token2index, token_parser=token_parser)
        self.exclude_roles = {
            bblfsh_roles.EXPRESSION,
            bblfsh_roles.IDENTIFIER,
            bblfsh_roles.LEFT,
            bblfsh_roles.QUALIFIED,
            bblfsh_roles.BINARY,
            bblfsh_roles.ASSIGNMENT,
        }

    def __call__(self, uast: bblfsh.Node) -> Iterable[Tuple[str, str]]:
        """
        Converts a UAST to a list of identifier, role pairs.
        The tokens are preprocessed by _token_parser.

        :param uast: The UAST root node.
        :return: a list of identifier, role pairs.
        """
        yield from self._process_uast(uast, [])

    def _process_uast(self, uast: bblfsh.Node, ancestors):
        stack = [(uast, [])]
        while stack:
            node, ancestors = stack.pop()

            if bblfsh_roles.IDENTIFIER in node.roles and node.token:
                roles = set(node.roles)
                indx = -1
                # We skip all Nodes with roles from `self.exclude_roles` set.
                # We skip any Node with OPERATOR role.
                # For them we take first parent Node from stack with another Role set.
                while not (roles - self.exclude_roles and bblfsh_roles.OPERATOR not in roles):
                    roles = set(ancestors[indx].roles)
                    indx -= 1
                for sub in self._token_parser.process_token(node.token):
                    try:
                        yield (self._token2index[sub], self.merge_roles(roles))
                    except KeyError:
                        continue
            ancestors = list(ancestors)
            ancestors.append(node)
            stack.extend([(child, ancestors) for child in node.children])

    @staticmethod
    def merge_roles(roles: Iterable[int]):
        return " | ".join(bblfsh.role_name(r) for r in sorted(roles))
