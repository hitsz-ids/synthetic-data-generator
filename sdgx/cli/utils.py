from functools import wraps

import click

from sdgx.cli.message import ExceptionMessage, NormalMessage
from sdgx.log import LOG_TO_FILE, add_log_file_handler, logger


def cli_wrapper(func):
    @click.option("--json_output", type=bool, default=False, help="Exit with json output.")
    @click.option("--log_to_file", type=bool, default=False, help="Log to file.")
    @wraps(func)
    def wrapper(json_output, log_to_file, *args, **kwargs):
        if log_to_file and not LOG_TO_FILE:
            add_log_file_handler()
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            if json_output:
                ExceptionMessage.from_exception(e).send()
            exit(getattr(e, "EXIT_CODE", -1))
        else:
            if json_output:
                NormalMessage().send()
            exit(0)

    return wrapper
