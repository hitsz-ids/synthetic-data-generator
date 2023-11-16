import click


@click.command()
def fit():
    print("sdgx fit")


@click.command()
def sample():
    print("sdgx sample")


@click.command()
def fit_and_sample():
    print("sdgx fit and sample")


@click.group()
def cli():
    pass


cli.add_command(fit)
cli.add_command(sample)
cli.add_command(fit_and_sample)
