from typing import Union, Dict
from pyspark.rdd import RDD
from pyspark import Row

from sourced.ml.transformers.transformer import Transformer
from sourced.ml.models import DocumentFrequencies


class Indexer(Transformer):
    """
    Maps each value of the given column in an RDD of pyspark.sql.Row to the respective integer
    index. The mapping is created by collecting all the unique values, sorting them and finally
    enumerating. Use value_to_index or [] to get index value.
    """
    def __init__(self, column: Union[int, str], column2id: Dict[str, int]=None, **kwargs):
        """
        :param column: column index or its name in pyspark.RDD for indexing.
        :param column2id: precomputed mapping between column values to integers.
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.column = column
        self._value_to_index = column2id
        self._values = None

    def __getitem__(self, key: Union[int, str]):
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

    def __len__(self):
        return len(self._value_to_index)

    @property
    def value_to_index(self):
        if self._value_to_index is None:
            raise AttributeError("column2id value not available. Run Indexer first.")
        return self._value_to_index

    def values(self):
        if self._values is None:
            self._values = [None] * len(self)
            for k, v in self.value_to_index.items():
                self._values[v] = k
        return self._values

    def calculate_value_to_index(self, rdd: RDD):
        column_name = self.column
        if isinstance(column_name, str):
            column = rdd.map(lambda x: getattr(x, column_name))
        else:
            column = rdd.map(lambda x: x[column_name])
        self._log.info("Collecting the list of distinct sorted values (%s)", column_name)
        values = column.distinct()
        if self.explained:
            self._log.info("toDebugString():\n%s", values.toDebugString().decode())
        values = values.collect()
        values.sort()  # We do not expect an extraordinary number of distinct values
        self._log.info("%d distinct values", len(values))
        if len(values) == 0:
            raise RuntimeError("Number of distinct values is zero.")
        self._value_to_index = {d: i for i, d in enumerate(values)}

    def save_index(self, path):
        self._log.info("Saving the index to %s", path)
        DocumentFrequencies().construct(len(self._value_to_index), self._value_to_index) \
            .save(path)

    def __call__(self, rdd: RDD):
        column_name = self.column
        if self._value_to_index is None:
            self.calculate_value_to_index(rdd)
        column2id = rdd.context.broadcast(self._value_to_index)

        def index_column(row):
            """
            Map column_id column to its index value stored in column2id.
            WARNING: due to pyspark documentation
            (http://spark.apache.org/docs/latest/rdd-programming-guide.html#passing-functions-to-spark)
            do not use self inside this function. It will be suboptimal and probably fail to run.
            Please contact me if you have troubles: kslavnov@gmail.com
            """
            if isinstance(column_name, str):
                assert isinstance(row, Row)
                row_dict = row.asDict()
                row_dict[column_name] = column2id.value[row_dict[column_name]]
                return Row(**row_dict)
            return row[:column_name] + (column2id.value[row[column_name]],) + row[column_name + 1:]
        indexed_rdd = rdd.map(index_column)
        column2id.unpersist(blocking=True)
        return indexed_rdd
