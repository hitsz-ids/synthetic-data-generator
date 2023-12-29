import json

import pytest

from sdgx.cli.message import ExceptionMessage, NormalMessage
from sdgx.exceptions import SdgxError


@pytest.mark.parametrize("return_val", [0, "123", [1, 2, 3], {"a": 1, "b": 2}])
def test_normal_message(return_val):
    NormalMessage.from_return_val(return_val)._dump_json == json.dumps(
        {
            "code": 0,
            "msg": "Success",
            "payload": return_val if isinstance(return_val, dict) else {"return_val": return_val},
        }
    )


def unknown_exception():
    raise Exception


def sdgx_exception():
    raise SdgxError


@pytest.mark.parametrize("exception_caller", [unknown_exception, sdgx_exception])
def test_exception_message(exception_caller):
    try:
        exception_caller()
    except Exception as e:
        msg = ExceptionMessage.from_exception(e)
        assert msg._dump_json()
        assert msg.code != 0
        assert msg.payload
        assert "details" in msg.payload


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
