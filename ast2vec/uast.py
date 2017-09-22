from modelforge import generate_meta
from modelforge.model import Model, split_strings, merge_strings, write_model
from modelforge.models import register_model

import ast2vec


@register_model
class UASTModel(Model):
    """
    Model to store Univeral Abstract Syntax Trees
    """
    NAME = "uast"

    def construct(self, repository, filenames, uasts):
        if len(uasts) != len(filenames):
            raise ValueError("Length of uasts({}) and filenames({}) are not equal".
                             format(len(uasts), len(filenames)))
        self._repository = repository
        self._filenames = filenames
        self._uasts = uasts
        self._filenames_map = {r: i for i, r in enumerate(self._filenames)}
        return self

    def _load_tree_kwargs(self, tree):
        return dict(repository=tree["repository"],
                    filenames=split_strings(tree["filenames"]),
                    uasts=[type(self).parse_bblfsh_response(uast)
                           for uast in split_strings(tree["uasts"])])

    @staticmethod
    def parse_bblfsh_response(response):
        # ParseResponse should be imported here because grpc starts threads during import
        # and if you call fork after that, a child process will be hang during exit
        from ast2vec.bblfsh_roles import Node
        return Node.FromString(response)

    def _load_tree(self, tree):
        self.construct(**self._load_tree_kwargs(tree))

    def dump(self):
        symbols_num = 100
        out = self._uasts[0][:symbols_num]
        return "Repository: %s\nNumber of files: %d. First %d symbols:\n %s" % (
            self._repo, len(self), symbols_num, out)

    @property
    def repository(self):
        """
        Returns the associated repository URL or path.
        """
        return self._repository

    @property
    def uasts(self):
        """
        Returns all uasts of code in the saved repo
        """
        return self._uasts

    @property
    def filenames(self):
        """
        Returns all filenames in the saved repo
        """
        return self._filenames

    def __getitem__(self, item):
        """
        Returns file name and uast for the given file index.

        :param item: File index.
        :return: name, uast
        """
        return self._filenames[item], self._uasts[item]

    def __iter__(self):
        """
        Iterator over the items.
        """
        return zip(self._filenames, self._uasts)

    def __len__(self):
        """
        Returns the number of files.
        """
        return len(self._filenames)

    def repository_index_by_name(self, name):
        """
        Looks up file index by it's name.
        """
        return self._filenames_map[name]

    def _to_dict_to_save(self):
        return {"repository": self.repository,
                "filenames": merge_strings(self.filenames),
                "uasts": merge_strings([uast.SerializeToString() for uast in self.uasts])}

    def save(self, output, deps=None):
        if not deps:
            deps = tuple()
        self._meta = generate_meta(self.NAME, ast2vec.__version__, *deps)
        write_model(self._meta, self._to_dict_to_save(), output)
