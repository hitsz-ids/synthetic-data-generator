from pydantic import BaseModel

from sdgx.exceptions import SdgxError


class ExitMessage(BaseModel):
    code: int
    msg: str
    payload: dict = {}

    def _dump_json(self) -> str:
        return self.model_dump_json()

    def send(self):
        print(self._dump_json(), flush=True, end="")


class NormalMessage(ExitMessage):
    code: int = 0
    msg: str = "Success"

    @classmethod
    def from_return_val(cls, return_val) -> "NormalMessage":
        if isinstance(return_val, dict):
            payload = return_val
        else:
            payload = {"return_val": return_val}
        return cls(code=0, msg="Success", payload=payload)


class ExceptionMessage(ExitMessage):
    @classmethod
    def from_exception(cls, e: Exception) -> "ExceptionMessage":
        if isinstance(e, SdgxError):
            return cls(
                code=e.ERROR_CODE,
                msg=str(e),
                payload={
                    "details": "Synthetic Data Generator Error, please check logs and raise an issue on https://github.com/hitsz-ids/synthetic-data-generator."
                },
            )
        return cls(
            code=-1,
            msg=str(e),
            payload={
                "details": "Unknown Exceptions, please check logs and raise an issue on https://github.com/hitsz-ids/synthetic-data-generator."
            },
        )
