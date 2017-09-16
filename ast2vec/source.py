from modelforge.model import split_strings, merge_strings
from modelforge.models import register_model

from ast2vec.uast import UASTModel


@register_model
class Source(UASTModel):
    """
    Model for source-code storage
    """
    NAME = "source"

    def construct(self, repository, filenames, uasts, sources):
        super().construct(repository=repository, filenames=filenames, uasts=uasts)
        if not len(sources) == len(uasts) == len(filenames):
            raise ValueError("Length of src_codes({}), uasts({}) and filenames({}) are not equal".
                             format(len(sources), len(uasts), len(filenames)))
        self._sources = sources
        self._filenames_map = {r: i for i, r in enumerate(self._filenames)}
        return self

    def _load_tree_kwargs(self, tree):
        tree_kwargs = super()._load_tree_kwargs(tree)
        tree_kwargs["sources"] = split_strings(tree["sources"])
        return tree_kwargs

    def dump(self):
        symbols_num = 100
        out = self._sources[0][:symbols_num]
        return "Number of files: %d. First %d symbols:\n %s" % (
            len(self._filenames), symbols_num, out)

    @property
    def sources(self):
        """
        Returns all code files of the saved repo
        """
        return self._sources

    def __getitem__(self, item):
        """
        Returns file name, source code and uast for the given file index.

        :param item: File index.
        :return: name, uast, source code
        """
        return super().__getitem__(item) + (self._sources[item],)

    def __iter__(self):
        """
        Iterator over the items.
        """
        return zip(self._filenames, self._uasts, self._sources)

    def _to_dict_to_save(self):
        save_dict = super()._to_dict_to_save()
        save_dict["sources"] = merge_strings(self.sources)
        return save_dict
