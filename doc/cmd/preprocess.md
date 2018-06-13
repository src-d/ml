# Preprocess command

This command allows you to preprocess your data before passing it to any command you need.
It converts the input files into Parquet files, after selecting commits, filtering by language, extracting UASTs and possibly taking out some of the fields.
You can specify the following arguments:

- `-r`/`--repositories` : Path to the input files
- `--parquet`: If your input files are Parquet files
- `--graph`: Path to the output Graphviz file if you wish to keep the tree
- `-o`/`--output`: Path to the output Parquet files
- `-x`/`--mode`: What to extract from repositories: files, functions or repository itself
- `-f`/`--fields`: Fields to keep, defaults to all, i.e. "blob_id", "repository_id", "content", "path", "commit_hash" and "uast"
- `-l`/`--languages` : Languages to keep, defaults to all languages detected by Babelfish
- `--dzhigurda`: Index of the last commit to keep, defaults to 0 (only the head), 1 is HEAD~1, etc.
- [Spark and Engine arguments](https://github.com/src-d/ml/blob/master/doc/spark.md)
