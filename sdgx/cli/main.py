from __future__ import annotations

import json
import time
from pathlib import Path

import click

from sdgx.cachers.manager import CacherManager
from sdgx.cli.utils import cli_wrapper
from sdgx.data_connectors.manager import DataConnectorManager
from sdgx.data_exporters.manager import DataExporterManager
from sdgx.data_processors.manager import DataProcessorManager
from sdgx.log import logger
from sdgx.models.manager import ModelManager
from sdgx.synthesizer import Synthesizer


@click.command()
@click.option(
    "--save_dir",
    type=str,
    required=True,
    default="",
    help="The directory to save the synthesizer",
)
@click.option(
    "--model",
    type=str,
    required=True,
    help="The name of the model.",
)
@click.option(
    "--model_path",
    type=str,
    default=None,
    help="The path of the model to load",
)
@click.option(
    "--model_kwargs",
    type=str,
    default=None,
    help="[Json String] The kwargs of the model for initialization",
)
@click.option(
    "--load_dir",
    type=str,
    default=None,
    help="The directory to load the synthesizer, if it is specified, ``model_path`` will be ignored.",
)
@click.option(
    "--metadata_path",
    type=str,
    default=None,
    help="The path of the metadata to load",
)
@click.option(
    "--data_connector",
    type=str,
    default=None,
    help="The name of the data connector to use",
)
@click.option(
    "--data_connector_kwargs",
    type=str,
    default=None,
    help="[Json String] The kwargs of the data connector to use",
)
@click.option(
    "--raw_data_loaders_kwargs",
    type=str,
    default=None,
    help="[Json String] The kwargs of the raw data loader to use",
)
@click.option(
    "--processed_data_loaders_kwargs",
    type=str,
    default=None,
    help="[Json String] The kwargs of the processed data loader to use",
)
@click.option(
    "--data_processors",
    type=str,
    default=None,
    help="[List str] The name of the data processors to use, e.g. 'processor_x,processor_y'",
)
@click.option(
    "--data_processors_kwargs",
    type=str,
    default=None,
    help="[Json String] The kwargs of the data processors to use",
)
@click.option(
    "--inspector_max_chunk",
    type=int,
    default=None,
    help="The max chunk of the inspector to load",
)
@click.option(
    "--metadata_include_inspectors",
    type=str,
    default=None,
    help="[List str] The name of the inspectors to include, e.g. 'inspector_x,inspector_y'",
)
@click.option(
    "--metadata_exclude_inspectors",
    type=str,
    default=None,
    help="[List str] The name of the inspectors to exclude, e.g. 'inspector_x,inspector_y'",
)
@click.option(
    "--inspector_init_kwargs",
    type=str,
    default=None,
    help="[Json String] The kwargs of the inspector to use",
)
@click.option(
    "--model_fit_kwargs",
    type=str,
    default=None,
    help="[Json String] The kwargs of the model fit method",
)
@click.option(
    "--dry_run",
    type=bool,
    default=False,
    help="Only init the synthesizer without fitting and save.",
)
@cli_wrapper
def fit(
    save_dir: str,
    model: str,
    model_path: str | None,
    model_kwargs: str | None,
    load_dir: str | None,
    metadata_path: str | None,
    data_connector: str | None,
    data_connector_kwargs: str | None,
    raw_data_loaders_kwargs: str | None,
    processed_data_loaders_kwargs: str | None,
    data_processors: str | None,
    data_processors_kwargs: str | None,
    # ``fit`` args
    inspector_max_chunk: int | None,
    metadata_include_inspectors: str | None,
    metadata_exclude_inspectors: str | None = None,
    inspector_init_kwargs: str | None = None,
    model_fit_kwargs: str | None = None,
    # Others
    dry_run: bool = False,
):
    """
    Fit the synthesizer or load a synthesizer for fitnuning/retraining/continue training...
    """
    if data_processors is not None:
        data_processors = data_processors.strip().split(",")

    if model_kwargs is not None:
        model_kwargs = json.loads(model_kwargs)
    if data_connector_kwargs is not None:
        data_connector_kwargs = json.loads(data_connector_kwargs)
    if raw_data_loaders_kwargs is not None:
        raw_data_loaders_kwargs = json.loads(raw_data_loaders_kwargs)
    if processed_data_loaders_kwargs is not None:
        processed_data_loaders_kwargs = json.loads(processed_data_loaders_kwargs)
    if data_processors_kwargs is not None:
        data_processors_kwargs = json.loads(data_processors_kwargs)
    if load_dir:
        if model_path:
            logger.warning(
                "Both ``model_path`` and ``load_dir`` are specified, ``model_path`` will be ignored."
            )
        synthesizer = Synthesizer.load(
            load_dir=load_dir,
            metadata_path=metadata_path,
            data_connector=data_connector,
            data_connector_kwargs=data_connector_kwargs,
            raw_data_loaders_kwargs=raw_data_loaders_kwargs,
            processed_data_loaders_kwargs=processed_data_loaders_kwargs,
            data_processors=data_processors,
            data_processors_kwargs=data_processors_kwargs,
        )
    else:
        if model_kwargs and model_path:
            logger.warning(
                "Both ``model_kwargs`` and ``model_path`` are specified, ``model_kwargs`` will be ignored."
            )
        synthesizer = Synthesizer(
            model=model,
            model_kwargs=model_kwargs,
            model_path=model_path,
            metadata_path=metadata_path,
            data_connector=data_connector,
            data_connector_kwargs=data_connector_kwargs,
            raw_data_loaders_kwargs=raw_data_loaders_kwargs,
            processed_data_loaders_kwargs=processed_data_loaders_kwargs,
            data_processors=data_processors,
            data_processors_kwargs=data_processors_kwargs,
        )
    if dry_run:
        return
    fit_kwargs = {}
    if inspector_max_chunk is not None:
        fit_kwargs["inspector_max_chunk"] = inspector_max_chunk
    if metadata_include_inspectors is not None:
        fit_kwargs["metadata_include_inspectors"] = metadata_include_inspectors.strip().split(",")
    if metadata_exclude_inspectors is not None:
        fit_kwargs["metadata_exclude_inspectors"] = metadata_exclude_inspectors.strip().split(",")
    if inspector_init_kwargs is not None:
        fit_kwargs["inspector_init_kwargs"] = json.loads(inspector_init_kwargs)
    if model_fit_kwargs is not None:
        fit_kwargs["model_fit_kwargs"] = json.loads(model_fit_kwargs)
    synthesizer.fit(**fit_kwargs)

    if not save_dir:
        save_dir = Path(f"./sdgx-fit-model-{model}-{time.time()}")
    else:
        save_dir = Path(save_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
    save_dir = synthesizer.save(save_dir)
    return save_dir.absolute().as_posix()


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
