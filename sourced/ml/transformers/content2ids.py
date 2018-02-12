import yaml

import pygments
from pygments.formatter import Formatter
from pygments.lexers import get_lexer_by_name, ClassNotFound

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

    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.linguist2pygments = {}
        self.split = args.split
        self.names = set()

    def __call__(self, rows):
        self.build_mapping()
        processed = rows.flatMap(self.process_row)
        if self.explained:
            self._log.info("toDebugString():\n%s", processed.toDebugString().decode())

        return processed \
            .distinct() \
            .map(lambda x: (x, " ".join(TokenParser().split(x))))

    def process_row(self, row):
        try:
            code = row.content
            try:
                lexer = get_lexer_by_name(self.linguist2pygments[row.lang][0])
            except ClassNotFound:
                lexer = get_lexer_by_name(self.linguist2pygments[row.lang][1])
            pygments.highlight(code, lexer, self.FormatterProxy(callback=self.process_tokens))
        except KeyError:
            pass
        for token in self.names:
            yield token

    def process_tokens(self, tokens):
        """
        Filter tokens of type "Name" and which are splittable
        according to :class: 'TokenParser' rules
        """
        for _type, token in tokens:
            if _type[0] == "Name":
                if self.split:
                    if len(list(TokenParser().split(token))) > 1:
                        self.names.add(token)
                else:
                    self.names.add(token)

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
