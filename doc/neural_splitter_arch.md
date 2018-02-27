# Neural Token Splitter architecture description

The goal of this feature is to train a neural network to learn how to split identifiers. Indeed, the rule-based `TokenParser` is not able to process some of them such as "`foobar`" or "`methodbase`". The main code is going to be stored in the algorithms directory: Class: `NeuralTokenSplitter`. The implementation plan is as follows.

1. Use `utils.engine.create_engine` to initialize a spark session and the engine with `args.repositories` as input, siva format. We plan to process the dataset of 150k top starred repositories in any languages. 

2. Code the new class named `CodeExtractor` in `transformers.basic` that is going to return both the source code as strings and their languages.

3. Get the Token stream of identifiers using [pygments.highlight](http://pygments.org/docs/quickstart/) with the `RawFormatter` and the language output by [enry](https://github.com/src-d/enry) as a lexer. Filter the token types with a callback. Design the code to easily switch between Babelfish and pygments.

4. Collect the training dataset

    * Select all identifiers splittable on special characters (`[^a-zA-Z]+`) or case changes. Ex `foo_bar` and `methodBase`

    * Use `TokenParser` from `algorithms.token_parser` to split them, make them lowercase, and join them again. Ex `foobar` and `methodbase`

    * We have X and Y:
        * `foobar` -> `foo bar`
        * `methodbase` -> `method base`

5. Code a simple neural language model with Keras that relies on character-level inputs as it is described in [Character-Aware Neural Language Models](https://arxiv.org/pdf/1508.06615.pdf) to split identifiers. 

6. Evaluation plots and accuracy improvments by playing with metaparameters. Other ideas:

    * Use the context of identifiers

    * Use [Babelfish](https://github.com/bblfsh) and focus on Python/Java languages
