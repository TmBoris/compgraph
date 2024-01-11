import json
import click
from compgraph.algorithms import pmi_graph


@click.command()
@click.argument('input_filepath', nargs=1)
@click.argument('output_filepath', nargs=1)
def pmi(input_filepath: str, output_filepath: str) -> None:
    graph = pmi_graph(name=input_filepath, file=True)

    result = graph.run()
    with open(output_filepath, "w") as out:
        json.dump((list(result)), out)


if __name__ == '__main__':
    pmi()
