import click

from sdgx.models.manager import ModelManager


@click.command()
def fit():
    print("sdgx fit")


@click.command()
def sample():
    print("sdgx sample")


@click.command()
def list_models():
    for model_name, model_cls in ModelManager().registed_model.items():
        print(f"{model_name} is registed as class: {model_cls}.")


@click.group()
def cli():
    pass


cli.add_command(fit)
cli.add_command(sample)
cli.add_command(list_models)
