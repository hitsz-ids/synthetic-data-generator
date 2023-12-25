import os

USER_DEFINED_LOG_LEVEL = os.getenv("SDGX_LOG_LEVEL", "INFO")
LOG_TO_FILE = os.getenv("SDGX_LOG_TO_FILE", "false") in ["True", "true"]

os.environ["LOGURU_LEVEL"] = USER_DEFINED_LOG_LEVEL

from loguru import logger


def add_log_file_handler():
    logger.add(
        "sdgx-{time}.log",
        rotation="10 MB",
    )


if LOG_TO_FILE:
    add_log_file_handler()

__all__ = ["logger", "LOG_TO_FILE", "add_log_file_handler", "USER_DEFINED_LOG_LEVEL"]
