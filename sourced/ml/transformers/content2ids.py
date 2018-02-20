import gzip
import operator
import os
import yaml

import pygments
from pygments.formatter import Formatter
from pygments.lexers import get_lexer_by_name, ClassNotFound
from pyspark import Row

from sourced.ml.algorithms import TokenParser
from sourced.ml.transformers import Transformer


class Content2Ids(Transformer):

    class FormatterProxy(Formatter):
        name = "Proxy"
        aliases = ["proxy"]
        filenames = []

        def __init__(self, **options):
            super(Content2Ids.FormatterProxy, self).__init__(**options)
            self.callback = options["callback"]

        def format(self, tokensource, outfile):
            self.callback(tokensource)

    def __init__(self, args, documents_column, **kwargs):
        super().__init__(**kwargs)
        self.documents_column = documents_column
        self.linguist2pygments = {}
        self.split = args.split
        self.idfreq = args.idfreq
        self.output = args.output

    def __call__(self, rows):
        list_RDDs = []
        self.build_mapping()
        processed = rows.flatMap(self._process_row).persist()

        if self.idfreq:
            num_repos_processed = processed \
                .map(lambda x: (x[0], x[1][0])) \
                .distinct()
            num_repos_reduced = self.reduce_rows(num_repos_processed)
            list_RDDs.append(num_repos_reduced)

            num_files_processed = processed \
                .map(lambda x: (x[0], x[1][1])) \
                .distinct()
            num_files_reduced = self.reduce_rows(num_files_processed)
            list_RDDs.append(num_files_reduced)

            num_occ_reduced = self.reduce_rows(processed)
            list_RDDs.append(num_occ_reduced)

            return processed \
                .map(lambda x: x[0]) \
                .context.union(list_RDDs) \
                .groupByKey() \
                .mapValues(list) \
                .map(lambda x: Row(
                    token=x[0],
                    token_split=" ".join(TokenParser(min_split_length=1).split(x[0])),
                    num_repos=x[1][0],
                    num_files=x[1][1],
                    num_occ=x[1][2]))
        else:
            return processed \
                .map(lambda x: x[0]) \
                .distinct() \
                .map(lambda x: Row(
                    token=x,
                    token_split=" ".join(TokenParser(min_split_length=1).split(x))))

    def reduce_rows(self, rows):
        return rows \
            .map(lambda x: (x[0], 1)) \
            .reduceByKey(operator.add)

    def _process_row(self, row):
        self.names = []
        repo_id = getattr(row, self.documents_column[0])
        file_id = getattr(row, self.documents_column[1])
        path = os.path.join(repo_id, file_id)
        code = row.content
        try:
            lexer = get_lexer_by_name(self.linguist2pygments[row.lang][0])
            pygments.highlight(code, lexer, self.FormatterProxy(callback=self.process_tokens))
        except ClassNotFound:
            lexer = get_lexer_by_name(self.linguist2pygments[row.lang][1])
            pygments.highlight(code, lexer, self.FormatterProxy(callback=self.process_tokens))
        except KeyError:
            pass
        for token in self.names:
            yield token, (repo_id, path)

    def process_tokens(self, tokens):
        """
        Filter tokens of type "Name" and which are splittable
        according to :class: 'TokenParser' rules
        """
        for _type, token in tokens:
            if _type[0] == "Name":
                if self.split:
                    if len(list(TokenParser(min_split_length=1).split(token))) > 1:
                        self.names.append(token)
                else:
                    self.names.append(token)

    def build_mapping(self):
        """
        Builds the mapping between linguist languages and pygments names for lexers.
        """
        with open("doc/languages.yml") as f:
            all_languages = yaml.load(f)

        linguist_langs = {}
        for lang, specs in all_languages.items():
            if specs["type"] == "programming":
                linguist_langs[lang] = (set(specs.get("aliases", []) + [lang.lower()]), specs)

        pygments_langs = set()
        for lexer in pygments.lexers.LEXERS.values():
            lang_declensions = [lang.lower() for lang in lexer[2]] + [lexer[1].lower()]
            pygments_langs |= set(lang_declensions)

        for lang in linguist_langs:
            lang_names = linguist_langs.get(lang, (set(),))[0]
            inter = list(lang_names.intersection(pygments_langs))
            if inter:
                self.linguist2pygments[lang] = inter

    def save(self, id_rdd):
        with gzip.open(self.output, "w") as g:
            columns_names = ["token", "token_split"]
            if self.idfreq:
                columns_names.extend(["num_repos", "num_files", "num_occ"])
            g.write(str.encode(",".join(columns_names).upper() + "\n"))
            for row in id_rdd.collect():
                row_dict = row.asDict()
                row_list = []
                for col in columns_names:
                    row_list.append(str(row_dict[col]))
                g.write(str.encode(",".join(row_list) + "\n"))
