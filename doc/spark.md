# How to use Spark with source{d} ml

This is a quick description about how to configure Spark when using ml. Given the source{d} `engine` uses Spark to load git repositories, this document also covers it's usage in the first section. For more information about how `engine` works, you can go check out the [repository](https://github.com/src-d/engine).

## Engine arguments:

- `--repository-format`: The `engine` expects you to point towards folders containing Siva files by default, however using this flag you can also choose to use regular git repositories with `standard` or bare git repositories with `bare`;
- `--bblfsh`: If you are not running the `Babelfish` daemon locally, for example if you are using a container, you can specify the server's address with this flag;
- `--explain`: PySpark optimizes the data pipeline under the hood, if you wish to see in what way you can print the execution plan with this flag;
- `--engine`: If you wish to provide a different version to Spark then the installed one, you can specify it with this flag.

## Spark arguments:

- `-s`/`--spark`: Specify Spark's master address, defaults to local[*] (your local spark with all available cores);
- `--package`: Shortcut to specify the name of any additional Spark package you may want to add to the workers;
- `--pause`: If you wish to not terminate the Spark session at the end (until you press `Enter`), useful to inspect the Spark UI;
- `--dep-zip`: If given, will zip dependencies, **should only be used in Cluster mode** (this flag might change soon);
- `-m`/`--memory`: Shortcut to specify values for the executor memory, the driver memory and the maximum driver result size, in this order. If you do not know what these values correspond to, the links in the next section will help;
- `--spark-local-dir`: Shortcut to specify the directory Spark uses to store data, defaults to `/tmp/spark`;
- `--spark-log-level`: Shortcut to specify the Spark log level, one of "ALL", "DEBUG", "ERROR", "FATAL", "INFO", "OFF", "TRACE" or "WARN", defaults to "WARN";
- `--persist`: Shortcut to enable caching, and specify the Spark storage level. Caching is done by using a `Cacher` transformer in the code, that just calls the Spark RDD `persist` method with the storage level specified by this flag, unless it has been omitted in which case it does nothing.

Here is a quick recap on the different spark storage levels:
```
Level                Space used  CPU time  In memory  On disk  Serialized
-------------------------------------------------------------------------
MEMORY_ONLY          High        Low       Y          N        N
MEMORY_ONLY_SER      Low         High      Y          N        Y
MEMORY_AND_DISK      High        Medium    Some       Some     Some
MEMORY_AND_DISK_SER  Low         High      Some       Some     Y
DISK_ONLY            Low         High      N          Y        Y
```

As you can see if you have a ton of RAM and relatively small data, "MEMORY_ONLY" is good, if you have a ton of data, enough space and a bad CPU, "DISK_ONLY" might be preferable. Tests are advised to find what works best, however when running on large data this flag should not be omitted, in order to avoid recomputing some of the transformations, which can be a lengthy process.

- `--config`: This flag allows you to directly specify any of the parameters of the [Spark Configuration](https://spark.apache.org/docs/2.2.0/configuration.html), the format is `key=value` (e.g. `--config spark.executor.memory=4G  spark.driver.memory=10G spark.driver.maxResultSize=4G`).

## Performance tuning

Most of the tuning that can be done on `ml` is Spark related, so here are some useful resources to learn how to do that:

- The top answer to [this StackOverflow question](https://stackoverflow.com/questions/37871194/how-to-tune-spark-executor-number-cores-and-executor-memory) explains how to tune all executor related parameters. It is not up to date regarding how to specify the number of executors, which Spark calculates with: `floor(spark.cores.max / spark.executor.cores)` (so you can just specify both parameters);
- Tuning driver parameters is a bit more tricky, given it really depends on the job you are running. You can check out answers to [this StackOverflow question](https://stackoverflow.com/questions/27181737/how-to-deal-with-executor-memory-and-driver-memory-in-spark) for more details;
- The `Repartionner` transformer is upcoming, however if you already wish to tune your data's partitioning, [this StackOverflow question](https://stackoverflow.com/questions/45704156/what-is-the-difference-between-spark-sql-shuffle-partitions-and-spark-default-pa) explains the difference between the two main parameters that you can use, and if you do not know why partitioning matters then [this may help](https://hackernoon.com/managing-spark-partitions-with-coalesce-and-repartition-4050c57ad5c4);
- [This talk](https://www.youtube.com/watch?v=WyfHUNnMutg) and the ones you can find from there is pretty useful if you plan on contributing to `ml`.