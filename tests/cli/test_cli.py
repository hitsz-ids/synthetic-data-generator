import pytest
from click.testing import CliRunner

from sdgx.cli.main import (
    fit,
    list_cachers,
    list_data_connectors,
    list_data_processors,
    list_exporters,
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
        list_exporters,
        list_models,
    ],
)
def test_list_extension_api(command, json_output):
    runner = CliRunner()
    result = runner.invoke(command, ["--json_output", json_output])

    assert result.exit_code == 0
    if json_output:
        assert NormalMessage().model_dump_json() in result.output
        assert NormalMessage().model_dump_json() == result.output.strip().split("\n")[-1]
    else:
        assert NormalMessage().model_dump_json() not in result.output


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
