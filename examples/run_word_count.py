import json
import click
from compgraph.algorithms import word_count_graph


@click.command()
@click.argument('input_filepath', nargs=1)
@click.argument('output_filepath', nargs=1)
def word_count(input_filepath: str, output_filepath: str) -> None:
    graph = word_count_graph(name=input_filepath, text_column='text', count_column='count', file=True)

    result = graph.run()
    with open(output_filepath, "w") as out:
        json.dump((list(result)), out)


if __name__ == '__main__':
    word_count()
