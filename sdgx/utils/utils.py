import contextlib
import logging

import numpy as np
import torch


# 添加增加的日志log
def get_log_file_handler(log_path):
    # 具体参数请参考：
    # https://docs.python.org/3/library/logging.handlers.html#timedrotatingfilehandler
    # 查路径请参考：
    # fileshandle.baseFilename
    fileshandler = logging.handlers.TimedRotatingFileHandler(
        log_path, when="W6", interval=5, backupCount=15, encoding="utf-8"
    )
    fileshandler.suffix = "%Y%m%d_%H%M%S.log"
    fileshandler.setLevel(logging.DEBUG)
    fmt_str = "%(asctime)s %(levelname)s %(filename)s[%(lineno)d] %(message)s"
    formatter = logging.Formatter(fmt_str)
    fileshandler.setFormatter(formatter)
    return fileshandler


# 引入自 ctgan
# 未来会根据业务需求优化修改
def random_state(function):
    """Set the random state before calling the function.

    Args:
        function (Callable):
            The function to wrap around.
    """

    def wrapper(self, *args, **kwargs):
        if self.random_states is None:
            return function(self, *args, **kwargs)

        else:
            with set_random_states(self.random_states, self.set_random_state):
                return function(self, *args, **kwargs)

    return wrapper


@contextlib.contextmanager
def set_random_states(random_state, set_model_random_state):
    """Context manager for managing the random state.

    Args:
        random_state (int or tuple):
            The random seed or a tuple of (numpy.random.RandomState, torch.Generator).
        set_model_random_state (function):
            Function to set the random state on the model.
    """
    original_np_state = np.random.get_state()
    original_torch_state = torch.get_rng_state()

    random_np_state, random_torch_state = random_state

    np.random.set_state(random_np_state.get_state())
    torch.set_rng_state(random_torch_state.get_state())

    try:
        yield
    finally:
        current_np_state = np.random.RandomState()
        current_np_state.set_state(np.random.get_state())
        current_torch_state = torch.Generator()
        current_torch_state.set_state(torch.get_rng_state())
        set_model_random_state((current_np_state, current_torch_state))

        np.random.set_state(original_np_state)
        torch.set_rng_state(original_torch_state)
