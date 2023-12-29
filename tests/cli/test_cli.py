import json

import pytest
from click.testing import CliRunner

from sdgx.cli.main import (
    fit,
    list_cachers,
    list_data_connectors,
    list_data_exporters,
    list_data_processors,
    list_models,
    sample,
)
from sdgx.cli.message import NormalMessage


@pytest.mark.parametrize("json_output", [True, False])
@pytest.mark.parametrize(
    "command",
    [
        list_cachers,
        list_data_connectors,
        list_data_processors,
        list_data_exporters,
        list_models,
    ],
)
def test_list_extension_api(command, json_output):
    runner = CliRunner()
    result = runner.invoke(command, ["--json_output", json_output])

    assert result.exit_code == 0
    if json_output:
        assert NormalMessage()._dump_json() in result.output
        assert NormalMessage()._dump_json() == result.output.strip().split("\n")[-1]
    else:
        assert NormalMessage()._dump_json() not in result.output


@pytest.mark.parametrize("model", ["CTGAN"])
@pytest.mark.parametrize("json_output", [True, False])
@pytest.mark.parametrize("torchrun", [False])
def test_fit_save_load_sample(
    model, demo_single_table_path, cacher_kwargs, json_output, torchrun, tmp_path
):
    runner = CliRunner()
    save_dir = tmp_path / f"unittest-{model}"
    result = runner.invoke(
        fit,
        [
            "--save_dir",
            save_dir,
            "--model",
            model,
            "--model_kwargs",
            json.dumps({"epochs": 1}),
            "--data_connector",
            "csvconnector",
            "--data_connector_kwargs",
            json.dumps({"path": demo_single_table_path}),
            "--raw_data_loaders_kwargs",
            json.dumps({"cacher_kwargs": cacher_kwargs}),
            "--processed_data_loaders_kwargs",
            json.dumps({"cacher_kwargs": cacher_kwargs}),
            "--json_output",
            json_output,
            "--torchrun",
            torchrun,
        ],
    )

    assert result.exit_code == 0
    assert save_dir.exists()
    assert len(list(save_dir.iterdir())) > 0

    if json_output:
        assert json.loads(result.output.strip().split("\n")[-1])

    export_dst = tmp_path / "exported.csv"
    result = runner.invoke(
        sample,
        [
            "--load_dir",
            save_dir,
            "--model",
            model,
            "--json_output",
            json_output,
            "--export_dst",
            export_dst.as_posix(),
            "--torchrun",
            torchrun,
        ],
    )

    assert result.exit_code == 0
    assert export_dst.exists()
    if json_output:
        assert json.loads(result.output.strip().split("\n")[-1])


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
