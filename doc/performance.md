**Usage recommendation**

Most of the tuning that can be done on `ml` is spark related. There are currently 3 flags that can help you tune Spark. 

- The first is the `--persist` flag, which enables caching. Caching is done by using a `Cacher` transformer in the code, that just calls the spark persist method with the storage level specified by this flag, unless it has been omitted in which case it does nothing. It can be useful to enable caching in order to avoid recomputing some of the transformations.

Here is a quick recap of the different spark storage levels:
```
Level                Space used  CPU time  In memory  On disk  Serialized
-------------------------------------------------------------------------
MEMORY_ONLY          High        Low       Y          N        N
MEMORY_ONLY_SER      Low         High      Y          N        Y
MEMORY_AND_DISK      High        Medium    Some       Some     Some
MEMORY_AND_DISK_SER  Low         High      Some       Some     Y
DISK_ONLY            Low         High      N          Y        Y
```

As you can see if you have a ton of RAM and relatively small data, MEMORY_ONLY is good, if you have a ton of data, enough space and a bad CPU, DISK_ONLY might be preferable. Tests are advised to find what works best, in all cases when running on large data this flag should not be omitted, whatever the storage level chosen.


- The second way is the `--memory` flag, which allows you to specify values for (in this order) executor memory, driver memory and max driver result size. If you do not know what these values correspond to, the links in the next section will help.

- The last way is by using the `--config` flag, wich allows you to specify values for any of the spark parameters. [This](https://stackoverflow.com/questions/37871194/how-to-tune-spark-executor-number-cores-and-executor-memory) is a pretty useful guide to understand/tune everything related to executors (not up to date regarding how to specify the number of executors desired, calculated by `floor(spark.cores.max / spark.executor.cores)` ). You might also want to look into these parameters: `spark.sql.shuffle.partitions` and `spark.default.parallelism` to tune the partionning of your data after shuffle operations. [This](https://stackoverflow.com/questions/45704156/what-is-the-difference-between-spark-sql-shuffle-partitions-and-spark-default-pa) and [this](https://hackernoon.com/managing-spark-partitions-with-coalesce-and-repartition-4050c57ad5c4) can be useful to understand why it matters. Tuning the driver parameters is a bit more tricky, [here](https://stackoverflow.com/questions/27181737/how-to-deal-with-executor-memory-and-driver-memory-in-spark) given it really depends on what part of `ml` you need.

