import typing
from pyspark.rdd import RDD
from pyspark import Row

from sourced.ml.transformers.transformer import Transformer


class Indexer(Transformer):
    """
    Maps each value of the given column in an RDD of pyspark.sql.Row to the respective integer
    index. The mapping is created by collecting all the unique values, sorting them and finally
    enumerating. Use value_to_index or [] to get index value.
    """
    def __init__(self, column: typing.Union[int, str], **kwargs):
        """
        :param column: column index or its name in pyspark.RDD for indexing.
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.column = column
        self._value_to_index = None
        self._values = None

    def __getitem__(self, key: typing.Union[int, str]):
        """
        Get index for given value or value for given index.
        If key type is string, __getitem__ gives you index value for key.
        If key type is int, __getitem__ gives you value for this index.
        WARNING: Do not use this function if you index non-string values, use `value_to_index` and
        values properties instead.

        :param key: value or index.
        :return: index for given value or value for given index.
        """
        if not isinstance(key, (int, str)):
            raise TypeError("__getitem__ supports only for string and int types. Use "
                            "`value_to_index` property to get index value.")
        return self.value_to_index[key] if isinstance(key, str) else self.values[key]

    @property
    def value_to_index(self):
        if self._value_to_index is None:
            raise AttributeError("column2id value not available. Run Indexer first.")
        return self._value_to_index

    @property
    def values(self):
        if self._values is None:
            raise AttributeError("column_values value not available. Run Indexer first.")
        return self._values

    def __call__(self, rdd: RDD):
        column_id = self.column
        if isinstance(column_id, str):
            column = rdd.map(lambda x: getattr(x, column_id))
        else:
            column = rdd.map(lambda x: x[column_id])

        self._log.info("Collecting the list of distinct sorted values (%s)", column_id)
        self._values = column \
            .sortBy(lambda x: x) \
            .distinct() \
            .collect()
        self._log.info("Done")

        column2id = {d: i for i, d in enumerate(self._values)}
        self._value_to_index = column2id

        def index_column(row):
            """
            Map column_id column to its index value stored in column2id.
            WARNING: due to pyspark documentation
            (http://spark.apache.org/docs/latest/rdd-programming-guide.html#passing-functions-to-spark)
            do not use self inside this function. It will be suboptimal and probably fail to run.
            Please contact me if you have troubles: kslavnov@gmail.com
            """
            if isinstance(column_id, str):
                assert isinstance(row, Row)
                row_dict = row.asDict()
                row_dict[column_id] = column2id[row_dict[column_id]]
                return Row(**row_dict)
            return row[:column_id] + (column2id[row[column_id]],) + row[column_id + 1:]

        return rdd.map(index_column)
