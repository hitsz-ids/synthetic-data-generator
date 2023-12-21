from __future__ import annotations

import json
from pathlib import Path

import click
import pandas

from sdgx.cachers.manager import CacherManager
from sdgx.cli.utils import cli_wrapper
from sdgx.data_connectors.manager import DataConnectorManager
from sdgx.data_exporters.manager import DataExporterManager
from sdgx.data_processors.manager import DataProcessorManager
from sdgx.models.manager import ModelManager


@click.command()
@click.option(
    "--model",
    help="Name of model, use `sdgx list-models` to list all available models.",
    required=True,
)
@click.option(
    "--model_params",
    default="{}",
    help="[Json-string] Parameters for model.",
)
@click.option(
    "--input_path",
    help="Path of input data.",
    required=True,
)
@click.option(
    "--input_type",
    default="csv",
    help="Type of input data, will be used as `pandas.read_{input_type}`.",
)
@click.option(
    "--read_params",
    default="{}",
    help="[Json-string] Parameters for `pandas.read_{input_type}`.",
)
@click.option(
    "--fit_params",
    default="{}",
    help="[Json-string] Parameters for `model.fit`.",
)
@click.option(
    "--output_path",
    help="Path to save the model.",
    required=True,
)
@cli_wrapper
def fit(
    model,
    model_params,
    input_path,
    input_type,
    read_params,
    fit_params,
    output_path,
):
    model_params = json.loads(model_params)
    read_params = json.loads(read_params)
    fit_params = json.loads(fit_params)

    model = ModelManager().init_model(model, **model_params)
    input_method = getattr(pandas, f"read_{input_type}")
    if not input_method:
        raise NotImplementedError(f"Pandas not support read_{input_type}")
    df = input_method(input_path, **read_params)
    model.fit(df, **fit_params)

    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save(output_path)


@click.command()
@cli_wrapper
def sample(
    model_path,
    output_path,
    write_type,
    write_param,
):
    model = ModelManager.load(model_path)
    # TODO: Model not have `sample` in Base Class yet
    # sampled_data = model.sample()
    # write_method = getattr(sampled_data, f"to_{write_type}")
    # write_method(output_path, write_param)


@click.command()
@cli_wrapper
def list_models():
    for model_name, model_cls in ModelManager().registed_models.items():
        print(f"{model_name} is registed as class: {model_cls}.")


@click.command()
@cli_wrapper
def list_data_connectors():
    for (
        model_name,
        model_cls,
    ) in DataConnectorManager().registed_data_connectors.items():
        print(f"{model_name} is registed as class: {model_cls}.")


@click.command()
@cli_wrapper
def list_data_processors():
    for (
        model_name,
        model_cls,
    ) in DataProcessorManager().registed_data_processors.items():
        print(f"{model_name} is registed as class: {model_cls}.")


@click.command()
@cli_wrapper
def list_cachers():
    for model_name, model_cls in CacherManager().registed_cachers.items():
        print(f"{model_name} is registed as class: {model_cls}.")


@click.command()
@cli_wrapper
def list_exporters():
    for model_name, model_cls in DataExporterManager().registed_exporters.items():
        print(f"{model_name} is registed as class: {model_cls}.")


@click.group()
def cli():
    pass


cli.add_command(fit)
cli.add_command(sample)
cli.add_command(list_models)
cli.add_command(list_data_connectors)
cli.add_command(list_data_processors)
cli.add_command(list_cachers)
cli.add_command(list_exporters)


if __name__ == "__main__":
    cli()
