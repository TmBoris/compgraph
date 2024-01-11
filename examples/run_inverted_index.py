import json
import click
import os
from compgraph.algorithms import inverted_index_graph


@click.command()
@click.argument('input_filepath')
@click.argument('output_filepath')
def tf_idf(input_filepath: str, output_filepath: str) -> None:
    input_filepath = os.path.abspath(input_filepath)  # or Path(input_filepath).resolve()
    output_filepath = os.path.abspath(output_filepath)  # or Path(output_filepath).resolve()

    graph = inverted_index_graph(input_filepath, file=True)

    result = graph.run()
    with open(output_filepath, "w") as out:
        json.dump((list(result)), out)


if __name__ == '__main__':
    tf_idf()
