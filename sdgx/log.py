import os

USER_DEFINED_LOG_LEVEL = os.getenv("SDGX_LOG_LEVEL", "INFO")
LOG_TO_FILE = os.getenv("SDGX_LOG_TO_FILE", "false") in ["True", "true"]

os.environ["LOGURU_LEVEL"] = USER_DEFINED_LOG_LEVEL

from loguru import logger

if LOG_TO_FILE:
    logger.add(
        "sdgx-{time}.log",
        rotation="10 MB",
    )

__all__ = ["logger"]
