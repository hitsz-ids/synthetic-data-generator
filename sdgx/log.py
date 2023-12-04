import os

USER_DEFINED_LOG_LEVEL = os.getenv("SDGX_LOG_LEVEL", "INFO")

os.environ["LOGURU_LEVEL"] = USER_DEFINED_LOG_LEVEL

from loguru import logger
