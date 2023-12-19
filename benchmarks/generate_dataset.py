import datetime
import itertools
import random
import string
from pathlib import Path

import click

_HERE = Path(__file__).parent


def random_string(length):
    return "".join(random.choice(string.ascii_lowercase) for i in range(length))


def random_float():
    return random.random() * 1000


def random_int():
    return random.randint(0, 1000)


def random_timestamp():
    current_timestamp = datetime.datetime.now().timestamp()
    return current_timestamp + random_int()


def random_datetime():
    return datetime.datetime.fromtimestamp(random_timestamp())


@click.option(
    "--output_file",
    default=(_HERE / "dataset/benchmark.csv").as_posix(),
)
@click.option("--num_rows", default=1_000_000)
@click.option("--int_cols", default=15)
@click.option("--float_cols", default=15)
@click.option("--string_cols", default=10)
@click.option("--string_discrete_nums", default=50)
@click.option("--timestamp_cols", default=10)
@click.option("--datetime_cols", default=0)
@click.command()
def generate_dateset(
    output_file,
    num_rows,
    int_cols,
    float_cols,
    string_cols,
    string_discrete_nums,
    timestamp_cols,
    datetime_cols,
):
    headers = itertools.chain.from_iterable(
        [
            (f"int_col{i}" for i in range(int_cols)),
            (f"float_col{i}" for i in range(float_cols)),
            (f"string_col{i}" for i in range(string_cols)),
            (f"timestamp_col{i}" for i in range(timestamp_cols)),
            (f"datetime_col{i}" for i in range(datetime_cols)),
        ]
    )
    output_file = Path(output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    random_str_list = [random_string(25) for i in range(string_discrete_nums)]

    def _generate_one_line():
        return ",".join(
            map(
                str,
                itertools.chain(
                    (random_int() for _ in range(int_cols)),
                    (random_float() for _ in range(float_cols)),
                    (random.choice(random_str_list) for _ in range(string_cols)),
                    (random_timestamp() for _ in range(timestamp_cols)),
                    (random_datetime() for _ in range(datetime_cols)),
                ),
            )
        )

    with output_file.open("w") as f:
        f.write(",".join(headers) + "\n")

        chunk_size = 1000
        for i in range(0, num_rows, chunk_size):
            f.write("\n".join(_generate_one_line() for _ in range(chunk_size)))
            f.write("\n")


if __name__ == "__main__":
    generate_dateset()
