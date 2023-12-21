from pydantic import BaseModel

from sdgx.exceptions import SdgxError


class ExitMessage(BaseModel):
    code: int
    msg: str

    def send(self):
        print(self.model_dump_json(), flush=True)


class NormalMessage(ExitMessage):
    code: int = 0
    msg: str = "Success"


class ExceptionMessage(ExitMessage):
    @classmethod
    def from_exception(cls, e: Exception) -> "ExceptionMessage":
        if isinstance(e, SdgxError):
            return cls(code=e.EXIT_CODE, msg=e.msg)
        return cls(code=-1, msg=str(e))
