"""CLI."""

import argparse

from sdgx.models.components.sdv_ctgan.data import read_csv, read_tsv, write_tsv
from sdgx.models.components.sdv_ctgan.synthesizers.ctgan import CTGAN


def _parse_args():
    parser = argparse.ArgumentParser(description="CTGAN Command Line Interface")
    parser.add_argument("-e", "--epochs", default=300, type=int, help="Number of training epochs")
    parser.add_argument(
        "-t", "--tsv", action="store_true", help="Load data in TSV format instead of CSV"
    )
    parser.add_argument(
        "--no-header",
        dest="header",
        action="store_false",
        help="The CSV file has no header. Discrete columns will be indices.",
    )

    parser.add_argument("-m", "--metadata", help="Path to the metadata")
    parser.add_argument(
        "-d", "--discrete", help="Comma separated list of discrete columns without whitespaces."
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        help="Number of rows to sample. Defaults to the training data size",
    )

    parser.add_argument(
        "--generator_lr", type=float, default=2e-4, help="Learning rate for the generator."
    )
    parser.add_argument(
        "--discriminator_lr", type=float, default=2e-4, help="Learning rate for the discriminator."
    )

    parser.add_argument(
        "--generator_decay", type=float, default=1e-6, help="Weight decay for the generator."
    )
    parser.add_argument(
        "--discriminator_decay", type=float, default=0, help="Weight decay for the discriminator."
    )

    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="Dimension of input z to the generator."
    )
    parser.add_argument(
        "--generator_dim",
        type=str,
        default="256,256",
        help="Dimension of each generator layer. " "Comma separated integers with no whitespaces.",
    )
    parser.add_argument(
        "--discriminator_dim",
        type=str,
        default="256,256",
        help="Dimension of each discriminator layer. "
        "Comma separated integers with no whitespaces.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=500, help="Batch size. Must be an even number."
    )
    parser.add_argument(
        "--save", default=None, type=str, help="A filename to save the trained synthesizer."
    )
    parser.add_argument(
        "--load", default=None, type=str, help="A filename to load a trained synthesizer."
    )

    parser.add_argument(
        "--sample_condition_column", default=None, type=str, help="Select a discrete column name."
    )
    parser.add_argument(
        "--sample_condition_column_value",
        default=None,
        type=str,
        help="Specify the value of the selected discrete column.",
    )

    parser.add_argument("data", help="Path to training data")
    parser.add_argument("output", help="Path of the output file")

    return parser.parse_args()


def main():
    """CLI."""
    args = _parse_args()
    if args.tsv:
        data, discrete_columns = read_tsv(args.data, args.metadata)
    else:
        data, discrete_columns = read_csv(args.data, args.metadata, args.header, args.discrete)

    if args.load:
        model = CTGAN.load(args.load)
    else:
        generator_dim = [int(x) for x in args.generator_dim.split(",")]
        discriminator_dim = [int(x) for x in args.discriminator_dim.split(",")]
        model = CTGAN(
            embedding_dim=args.embedding_dim,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=args.generator_lr,
            generator_decay=args.generator_decay,
            discriminator_lr=args.discriminator_lr,
            discriminator_decay=args.discriminator_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )
    model.fit(data, discrete_columns)

    if args.save is not None:
        model.save(args.save)

    num_samples = args.num_samples or len(data)

    if args.sample_condition_column is not None:
        assert args.sample_condition_column_value is not None

    sampled = model.sample(
        num_samples, args.sample_condition_column, args.sample_condition_column_value
    )

    if args.tsv:
        write_tsv(sampled, args.metadata, args.output)
    else:
        sampled.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
