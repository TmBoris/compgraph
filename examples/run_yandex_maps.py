import json
import click
from compgraph.algorithms import yandex_maps_graph


@click.command()
@click.argument('first_input_filepath', nargs=1)
@click.argument('second_input_filepath', nargs=1)
@click.argument('output_filepath', nargs=1)
def yandex_maps(first_input_filepath: str, second_input_filepath: str, output_filepath: str) -> None:
    graph = yandex_maps_graph(name1=first_input_filepath, name2=second_input_filepath, file=True)

    result = graph.run()
    with open(output_filepath, "w") as out:
        json.dump((list(result)), out)


if __name__ == '__main__':
    yandex_maps()
