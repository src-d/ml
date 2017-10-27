from ast2vec.pickleable_logger import PickleableLogger  # nopep8


class Repo2Base(PickleableLogger):
    def convert_uast(self, uast):
        return self.convert_uasts([uast])

    def convert_uasts(self, file_uast_generator):
        raise NotImplementedError()
