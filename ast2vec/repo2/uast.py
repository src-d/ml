from ast2vec.repo2.base import Repo2Base, RepoTransformer, repos2_entry, repo2_entry
from ast2vec.uast import UASTModel


class Repo2UASTModel(Repo2Base):
    """
    Extract UASTs from repository
    """
    MODEL_CLASS = UASTModel

    def convert_uasts(self, file_uast_generator):
        """
        Collect filenames and UASTs from file_uast_generator.
        """
        uasts = []
        filenames = []

        for file_uast in file_uast_generator:
            uasts.append(file_uast.response.uast)
            filenames.append(file_uast.filename)

        return filenames, uasts


class Repo2UASTModelTransformer(RepoTransformer):
    WORKER_CLASS = Repo2UASTModel

    def dependencies(self):
        """
        Return the list of parent models which were used to generate the target one.
        """
        return []

    def result_to_model_kwargs(self, result, url_or_path):
        """
        Convert the "result" object from parse_uasts() to WORKER_CLASS.MODEL_CLASS.construct()
        keyword arguments.

        :param result: The object returned from parse_uasts().
        :param url_or_path: The repository's source.
        :return: :class:`dict` with the required items to construct the model.
        """
        filenames, uasts = result
        if len(filenames) == 0:
            raise ValueError("No need to store empty model.")
        return {"repository": url_or_path, "filenames": filenames, "uasts": uasts}


def repo2uast_entry(args):
    return repo2_entry(args, Repo2UASTModelTransformer)


def repos2uast_entry(args):
    return repos2_entry(args, Repo2UASTModelTransformer)
