from datetime import datetime
from abc import abstractmethod, ABC
import typing as tp
import string
from itertools import groupby
import heapq
import numpy as np
from collections import defaultdict


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))


TRow = dict[str, tp.Any]
TRowsIterable = tp.Iterable[TRow]
TRowsGenerator = tp.Generator[TRow, None, None]


class Operation(ABC):
    @abstractmethod
    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        pass


class AddOperation(Operation):
    def __init__(self, first: Operation, second: Operation) -> None:
        self.first = first
        self.second = second

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        yield from self.first(self.second(**kwargs))


class Read(Operation):
    def __init__(self, filename: str, parser: tp.Callable[[str], TRow]) -> None:
        self.filename = filename
        self.parser = parser

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        with open(self.filename) as f:
            for line in f:
                parsed = self.parser(line)
                if isinstance(parsed, list):  # type: ignore
                    yield from self.parser(line)  # type: ignore
                else:
                    yield self.parser(line)


class ReadIterFactory(Operation):
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in kwargs[self.name]():
            yield row


# Operations


class Mapper(ABC):
    """Base class for mappers"""

    @abstractmethod
    def __call__(self, row: TRow) -> TRowsGenerator:
        """
        :param row: one table row
        """
        pass


class Map(Operation):
    def __init__(self, mapper: Mapper) -> None:
        self.mapper = mapper

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in rows:
            res = self.mapper(row)
            for el in res:
                yield el


class Reducer(ABC):
    """Base class for reducers"""

    @abstractmethod
    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        """
        :param rows: table rows
        """
        pass


class Reduce(Operation):
    def __init__(self, reducer: Reducer, keys: tp.Sequence[str]) -> None:
        self.reducer = reducer
        self.keys = keys

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for key, group in groupby(rows, key=lambda row: [row[key] for key in self.keys]):
            for el in self.reducer(tuple(self.keys), group):
                yield el


class Joiner(ABC):
    """Base class for joiners"""

    def __init__(self, suffix_a: str = '_1', suffix_b: str = '_2') -> None:
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b

    def get_ans(self, row_a: dict[str, tp.Any],
                row_b: dict[str, tp.Any], keys: tp.Sequence[str]) -> dict[str, tp.Any]:
        ans = {key: row_a[key] for key in keys}
        ans.update({key + self._a_suffix if key in row_b else key: value
                    for key, value in row_a.items() if key not in keys})
        ans.update({key + self._b_suffix if key in row_a else key: value
                    for key, value in row_b.items() if key not in keys})
        return ans

    @abstractmethod
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        """
        :param keys: join keys
        :param rows_a: left table rows
        :param rows_b: right table rows
        """
        pass


class Join(Operation):
    def __init__(self, joiner: Joiner, keys: tp.Sequence[str]):
        self.keys = keys
        self.joiner = joiner

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        grouped_rows_a = groupby(rows, key=lambda x: [x[key] for key in self.keys])
        grouped_rows_b = groupby(args[0], key=lambda x: [x[key] for key in self.keys])

        (key_a, group_a) = next(grouped_rows_a, (None, None))
        (key_b, group_b) = next(grouped_rows_b, (None, None))

        while key_a is not None and key_b is not None and group_a is not None and group_b is not None:
            if key_a == key_b:
                for el in self.joiner(self.keys, group_a, group_b):
                    yield el
                (key_a, group_a) = next(grouped_rows_a, (None, None))
                (key_b, group_b) = next(grouped_rows_b, (None, None))

            elif key_a < key_b:
                if isinstance(self.joiner, LeftJoiner | OuterJoiner):
                    for el in self.joiner(self.keys, group_a, []):
                        yield el
                (key_a, group_a) = next(grouped_rows_a, (None, None))

            else:
                if isinstance(self.joiner, RightJoiner | OuterJoiner):
                    for el in self.joiner(self.keys, [], group_b):
                        yield el
                (key_b, group_b) = next(grouped_rows_b, (None, None))

        while group_a is not None:
            if isinstance(self.joiner, LeftJoiner | OuterJoiner):
                for el in self.joiner(self.keys, group_a, []):
                    yield el
            (key_a, group_a) = next(grouped_rows_a, (None, None))

        while group_b is not None:
            if isinstance(self.joiner, RightJoiner | OuterJoiner):
                for el in self.joiner(self.keys, [], group_b):
                    yield el
            (key_b, group_b) = next(grouped_rows_b, (None, None))


# Dummy operators


class DummyMapper(Mapper):
    """Yield exactly the row passed"""

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield row


class FirstReducer(Reducer):
    """Yield only first row from passed ones"""

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        for row in rows:
            yield row
            break


# Mappers


class FilterPunctuation(Mapper):
    """Left only non-punctuation symbols"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = remove_punctuation(row[self.column])
        yield row


class LowerCase(Mapper):
    """Replace column value with value in lower case"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    @staticmethod
    def _lower_case(txt: str) -> str:
        return txt.lower()

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = self._lower_case(row[self.column])
        yield row


class Split(Mapper):
    """Split row on multiple rows by separator"""

    def __init__(self, column: str, separator: str | None = None) -> None:
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self.column = column
        self.separator = separator

    def __call__(self, row: TRow) -> TRowsGenerator:
        values = row[self.column].split()
        for value in values:
            new_row = row.copy()
            new_row[self.column] = value
            yield new_row


class Product(Mapper):
    """Calculates product of multiple columns"""

    def __init__(self, columns: tp.Sequence[str], result_column: str = 'product') -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self.columns = columns
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.result_column] = 1
        for column in self.columns:
            row[self.result_column] *= row[column]
        yield row


class Filter(Mapper):
    """Remove records that don't satisfy some condition"""

    def __init__(self, condition: tp.Callable[[TRow], bool]) -> None:
        """
        :param condition: if condition is not true - remove record
        """
        self.condition = condition

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.condition(row):
            yield row


class Project(Mapper):
    """Leave only mentioned columns"""

    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: names of columns
        """
        self.columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield {key: value for key, value in row.items() if key in self.columns}


class IDF(Mapper):
    def __init__(self, columns: tp.Sequence[str], result_col: str = 'idf') -> None:
        """
        :param columns: names of columns
        :param result_col: name of idf column
        """
        self.columns = columns
        self.result_col = result_col

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.result_col] = np.log(row[self.columns[0]] / row[self.columns[1]])
        yield row


class PMI(Mapper):
    def __init__(self, columns: tp.Sequence[str], result_col: str = 'pmi') -> None:
        """
        :param columns: names of columns
        :param result_col: name of idf column
        """
        self.columns = columns
        self.result_col = result_col

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.result_col] = np.log(row[self.columns[0]] / row[self.columns[1]])
        yield row


class Reveal(Mapper):
    def __init__(self, column: str) -> None:
        """
        :param column: name of column to reveal
        """
        self.column_to_reveal = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        times = row[self.column_to_reveal]
        del row[self.column_to_reveal]
        for _ in range(times):
            yield row


class Inverse(Mapper):
    def __init__(self, column: str) -> None:
        """
        :param column: name of column to reveal
        """
        self.column_to_inverse = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column_to_inverse] = -row[self.column_to_inverse]
        yield row


class GetDuration(Mapper):
    def __init__(self, start_col: str, leave_col: str, res_col_name: str = 'duration') -> None:
        """
        :param start_col: name of column with start time
        :param leave_col: name of column with leave time
        :param res_col_name: name of column with result duration
        """
        self.start_col = start_col
        self.leave_col = leave_col
        self.res_col_name = res_col_name

    def __call__(self, row: TRow) -> TRowsGenerator:
        date_format = "%Y%m%dT%H%M%S.%f"
        start = datetime.strptime(row[self.start_col], date_format)
        leave = datetime.strptime(row[self.leave_col], date_format)
        time_delta = leave - start
        row[self.res_col_name] = time_delta.total_seconds() / 3600
        yield row


class GetWeekdayAndHour(Mapper):
    def __init__(self, enter_time_col: str, weekday_res_col: str, hour_res_col: str) -> None:
        """
        :param enter_time_col: name of column with start time
        :param weekday_res_col: name of column with weekday of the trip
        :param hour_res_col: name of column with hour of the trip
        """
        self.enter_time_col = enter_time_col
        self.weekday_res_col = weekday_res_col
        self.hour_res_col = hour_res_col

    def __call__(self, row: TRow) -> TRowsGenerator:
        date_format = "%Y%m%dT%H%M%S.%f"
        dt_object = datetime.strptime(row[self.enter_time_col], date_format)
        row[self.weekday_res_col] = dt_object.strftime("%a")
        row[self.hour_res_col] = dt_object.hour
        yield row


class GetHaversineDist(Mapper):
    def __init__(self, start: str, end: str, res_col_name: str = 'distance') -> None:
        """
                :param start: name of column with start coordinates
                :param end: name of column with end coordinates
                :param res_col_name: name of column with result distance
                """
        self.start = start
        self.end = end
        self.res_col_name = res_col_name

    def __call__(self, row: TRow) -> TRowsGenerator:
        lng1, lat1 = row[self.start]
        lng2, lat2 = row[self.end]
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        radius = 6373  # in km
        lat = lat2 - lat1
        lng = lng2 - lng1
        d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
        row[self.res_col_name] = 2 * radius * np.arcsin(np.sqrt(d))
        yield row


class GetAverageSpeed(Mapper):
    def __init__(self, dist: str, duration: str, res_col_name: str = 'speed') -> None:
        """
        :param dist: name of column with total distances of that time
        :param duration: name of column with total durations of that time
        :param res_col_name: name of column with result average speed
        """
        self.dist = dist
        self.duration = duration
        self.res_col_name = res_col_name

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.res_col_name] = row[self.dist] / row[self.duration]
        yield row


# Reducers


class TopN(Reducer):
    """Calculate top N by value"""

    def __init__(self, column: str, n: int) -> None:
        """
        :param column: column name to get top by
        :param n: number of top values to extract
        """
        self.column_max = column
        self.n = n

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        heap: list[tuple[tp.Any, list[tuple[str, tp.Any]]]] = []
        for row in rows:
            values = [(key, value) for key, value in row.items()]
            heapq.heappush(heap, (row[self.column_max], values))
            if len(heap) > self.n:
                heapq.heappop(heap)

        for key, values in heap:
            yield {key: value for key, value in values}


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""

    def __init__(self, words_column: str, result_column: str = 'tf') -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.result_column = result_column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        words_count: dict[str, int] = defaultdict(int)
        keys: dict[str, tp.Any] = {}
        summ = 0

        for row in rows:
            if not keys:
                keys = {key: row[key] for key in group_key}
            words_count[row[self.words_column]] += 1
            summ += 1

        for word, count in words_count.items():
            ans = {**keys, self.words_column: word, self.result_column: count / summ}
            yield ans


class Count(Reducer):
    """
    Count records by key
    Example for group_key=('a',) and column='d'
        {'a': 1, 'b': 5, 'c': 2}
        {'a': 1, 'b': 6, 'c': 1}
        =>
        {'a': 1, 'd': 2}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for result column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        ans: dict[str, tp.Any] = {}
        num_rows = 0
        for row in rows:
            if not ans:
                for key in group_key:
                    ans[key] = row[key]
            num_rows += 1
        ans[self.column] = num_rows
        yield ans


class Sum(Reducer):
    """
    Sum values aggregated by key
    Example for key=('a',) and column='b'
        {'a': 1, 'b': 2, 'c': 4}
        {'a': 1, 'b': 3, 'c': 5}
        =>
        {'a': 1, 'b': 5}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for sum column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        ans: dict[str, tp.Any] = {}
        col_sum = 0
        for row in rows:
            if not ans:
                for key in group_key:
                    ans[key] = row[key]
            col_sum += row[self.column]
        ans[self.column] = col_sum
        yield ans


# Joiners


class InnerJoiner(Joiner):
    """Join with inner strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        list_rows_b = list(rows_b)
        for row_a in rows_a:
            for row_b in list_rows_b:
                ans = self.get_ans(row_a, row_b, keys)
                yield ans


class OuterJoiner(Joiner):
    """Join with outer strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        list_rows_a = list(rows_a)
        list_rows_b = list(rows_b)
        if len(list_rows_b) == 0:
            for el in list_rows_a:
                yield el
        elif len(list_rows_a) == 0:
            for el in list_rows_b:
                yield el
        else:
            for row_a in list_rows_a:
                for row_b in list_rows_b:
                    ans = self.get_ans(row_a, row_b, keys)
                    yield ans


class LeftJoiner(Joiner):
    """Join with left strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        list_rows_b = list(rows_b)
        if len(list_rows_b) == 0:
            for el in rows_a:
                yield el
        else:
            for row_a in rows_a:
                for row_b in list_rows_b:
                    ans = self.get_ans(row_a, row_b, keys)
                    yield ans


class RightJoiner(Joiner):
    """Join with right strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        list_rows_a = list(rows_a)
        if len(list_rows_a) == 0:
            for el in rows_b:
                yield el
        else:
            for row_b in rows_b:
                for row_a in list_rows_a:
                    ans = self.get_ans(row_a, row_b, keys)
                    yield ans
