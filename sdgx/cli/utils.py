import copy
import json
import re
import sys
from functools import wraps

import click
import importlib_metadata

from sdgx.cli.message import ExceptionMessage, NormalMessage
from sdgx.log import LOG_TO_FILE, add_log_file_handler, logger
from sdgx.utils import find_free_port


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


def load_entry_point(distribution, group, name):
    dist_obj = importlib_metadata.distribution(distribution)
    eps = [ep for ep in dist_obj.entry_points if ep.group == group and ep.name == name]
    if not eps:
        raise ImportError("Entry point %r not found" % ((group, name),))
    return eps[0].load()


def torch_run_warpper(func):
    """
    Experimental feature for native torchrun.

    Alternatively, people can use `torchrun $(which sdgx)`

    FIXME: This is not compatible with click.testing.CliRunner
    """

    @click.option("--torchrun", type=bool, default=False, help="Use `torchrun` to run cli.")
    @click.option(
        "--torchrun_kwargs",
        type=str,
        default="{}",
        help="[Json String] torchrun kwargs.",
    )
    @wraps(func)
    def wrapper(torchrun, torchrun_kwargs, *args, **kwargs):
        if not torchrun:
            func(*args, **kwargs)
        else:
            torchrun_kwargs = json.loads(torchrun_kwargs)
            torchrun_kwargs.setdefault("master_port", find_free_port())
            origin_args = copy.deepcopy(sys.argv)
            sys.argv = [re.sub(r"(-script\.pyw|\.exe)?$", "", sys.argv[0])]
            for k, v in torchrun_kwargs.items():
                sys.argv.extend([f"--{k}", str(v)])
            # Remove [--torchrun=true] and [--torchrun, true] from origin_args
            if "--torchrun" in origin_args:
                i = origin_args.index("--torchrun")
                if i + 1 < len(origin_args) and origin_args[i + 1] == "true":
                    origin_args.pop(i)
                    origin_args.pop(i)

            if "--torchrun=true" in origin_args:
                origin_args.remove("--torchrun=true")

            sys.argv.extend(origin_args)
            sys.exit(load_entry_point("torch", "console_scripts", "torchrun")())

    return wrapper
