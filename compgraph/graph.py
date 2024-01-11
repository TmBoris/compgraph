import typing as tp
import json

from . import operations as ops
from . import external_sort as ext_sort


class Graph:
    """Computational graph implementation"""
    def __init__(self) -> None:
        self.op: tp.Any = None

    @staticmethod
    def make_graph(input_stream_name: str, file: bool = False) -> 'Graph':
        if file:
            return Graph.graph_from_file(input_stream_name, json.loads)
        return Graph.graph_from_iter(input_stream_name)

    @staticmethod
    def graph_from_iter(name: str) -> 'Graph':
        """Construct new graph which reads data from row iterator (in form of sequence of Rows
        from 'kwargs' passed to 'run' method) into graph data-flow
        Use ops.ReadIterFactory
        :param name: name of kwarg to use as data source
        """
        graph = Graph()
        graph.op = ops.ReadIterFactory(name)
        return graph

    @staticmethod
    def graph_from_file(filename: str, parser: tp.Callable[[str], ops.TRow]) -> 'Graph':
        """Construct new graph extended with operation for reading rows from file
        Use ops.Read
        :param filename: filename to read from
        :param parser: parser from string to Row
        """
        graph = Graph()
        graph.op = ops.Read(filename, parser)
        return graph

    def map(self, mapper: ops.Mapper) -> 'Graph':
        """Construct new graph extended with map operation with particular mapper
        :param mapper: mapper to use
        """
        graph = Graph()
        graph.op = ops.AddOperation(ops.Map(mapper), self.op)
        return graph

    def reduce(self, reducer: ops.Reducer, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with reduce operation with particular reducer
        :param reducer: reducer to use
        :param keys: keys for grouping
        """
        graph = Graph()
        graph.op = ops.AddOperation(ops.Reduce(reducer, keys), self.op)
        return graph

    def sort(self, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with sort operation
        :param keys: sorting keys (typical is tuple of strings)
        """
        graph = Graph()
        graph.op = ops.AddOperation(ext_sort.ExternalSort(keys), self.op)
        return graph

    def join(self, joiner: ops.Joiner, join_graph: 'Graph', keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with join operation with another graph
        :param joiner: join strategy to use
        :param join_graph: other graph to join with
        :param keys: keys for grouping
        """
        graph = Graph()
        graph.op = GraphsJoiner(ops.Join(joiner, keys), self, join_graph)
        return graph

    def run(self, **kwargs: tp.Any) -> ops.TRowsIterable:
        """Single method to start execution; data sources passed as kwargs"""
        yield from self.op(**kwargs)


class GraphsJoiner(ops.Operation):
    """Join two graphs"""
    def __init__(self, joiner: ops.Join, g1: 'Graph', g2: 'Graph') -> None:
        self.joiner = joiner
        self.g1 = g1
        self.g2 = g2

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> ops.TRowsGenerator:
        yield from self.joiner(self.g1.run(**kwargs), self.g2.run(**kwargs))
