# Source{d} ml proposal-000
This document contains proposals and questions to discuss for [source{d} ml](https://github.com/src-d/ml).

## Package structure
Now it is a little bit confusing and flat. 
I suggest to have the following structure:

1. Transformers
	1. Keep here all transformers.
	2. Put small and common one to base.py
	3. Put big and independent to separate file
	4. Break wmhash.py into smaller pieces 
2. Extractors (can be inside transformers, but better to keep here)
	1. Keep all feature extractors here
3. Models
	3. Keep here all asdf models
4. Utils
	1. Move all helpers here like bigartm, engine, token_parser, vw_dataset, etc. 
	2. Can keep non-helpers outside: swivel, projector, …

## Package content
1. We should not have a bblfsh_meta.py with roles list and ask LA team to include it in bblfsh python package. 
2. We don’t need to have LazyGrpc anymore.

## Improvements
### 1. Roles and Node 
https://github.com/bblfsh/client-python/issues/34 and https://github.com/bblfsh/client-python/issues/35

Use Roles and Node from bblfsh and not like we do now
```
self.parse_uast = importlib.import_module(
   "bblfsh.gopkg.in.bblfsh.sdk.%s.uast.generated_pb2" % VERSION
```

### 2. Add bblfsh version check.
As Vadim ask here https://github.com/bblfsh/client-python/issues/50. Just not to forget.

### 3. What should be inside ml, and what on top of that?
From my point of view, `src-d/ml` is something to create final datasets and provide basic transformers for all ML models. 
The model training itself should be in a separate repository like Apollo.

Maybe it is a good idea to put a model in `sourced.ml` python subpackage when you install it.

Related questions:
1. Should we move swivel training outside?
2. Is everything in `wmhash.py` is common or we should move something to apollo repo?
3. Should we rename src-d ML one time more :trolling:?

### 4. Are bags everywhere?
We use “bag” term to denote “features” everywhere.
For me, bag denotes zero-one features, where you don’t have any order.
Usually, they are related to “word” information. 
The weighted bag is the same as features set, but more or less consistent. 

If we have some structural UAST features it is better not to call it bags. 
So in `uast_struct_to_bag.py` we have feature generation, not bags. 

### 5. `uast_ids_to_bag.py`, `uast_struct_to_bag.py`, `BagsExtractor`
We have transformers to transform data and have Bags(Features)Extractor to get features. 
It is ok and logical. But why do we have `UastIds2Bag` and `UastStructure2BagBase` with only one meaningful function: `uast_to_bag` (uast_to_features)? We don't use `vocabulary` property anymore. 

We have Bags(Features)Extractors as providers to this classes. They are pretty similar to each other. 
So let’s KISS and include a logical part to Bags(Features)Extractor.

### 6. `Uast_ids_to_bag.py` vs `token_mapper.py`
Related to 5.
Also, class  `UastIds2Bag` can be reimplemented using the engine.
What it does:
1. Takes all Nodes filtered by XPath rule (can do the same with engine )
```
uasts.query_uast('//*[@roleIdentifier and not(@roleIncomplete)]').extract_tokens()
```
2. Use `token_parser` to parse tokens
3. Use `_vocabulary` to map tokens to something as bag keys
4. Return bag keys frequencies.

Mostly the same we have in `token_mapper.py` using the engine (In PR right now).

### 7. Move `NoopTokenParser` and `HashedTokenParser` to parcers. And join with `NoTokenParser`.

### 8. Move `OrderedDocumentFrequencies` to models

### 9. Repo2DocFreq
Let’s include this step inside feature extraction because it just calculates docfrec independently for each feature set. Looks strange in the pipeline and not so beautiful inside.

Solutions
1. Include to `Repo2WeightedSet` Or to extractors itself
2. Calculate everything independently in the pipeline and Add special `JoinFeatures` Transformer to join all features by key to one rdd. It will be easier to understand what is doing on in the pipeline. 

The second way is preferable.

### 10. We don’t use VocabularyCooccurrences model anymore. Time to delete it.