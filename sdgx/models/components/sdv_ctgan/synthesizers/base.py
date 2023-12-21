"""BaseSynthesizer module."""

import contextlib

import cloudpickle
import numpy as np
import torch


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


class BaseSynthesizer:
    """Base class for all default synthesizers of ``CTGAN``.

    This should contain the save/load methods.
    """

    random_states = None

    def __getstate__(self):
        device_backup = self._device
        self.set_device(torch.device("cpu"))
        state = self.__dict__.copy()
        self.set_device(device_backup)

        random_states = self.random_states
        if (
            isinstance(random_states, tuple)
            and isinstance(random_states[0], np.random.RandomState)
            and isinstance(random_states[1], torch.Generator)
        ):
            state["_numpy_random_state"] = random_states[0].get_state()
            state["_torch_random_state"] = random_states[1].get_state()
            del state["random_states"]

        return state

    def __setstate__(self, state):
        np_state = state.pop("_numpy_random_state", None)
        torch_state = state.pop("_torch_random_state", None)
        if np_state is not None and torch_state is not None:
            current_torch_state = torch.Generator()
            current_torch_state.set_state(torch_state)
            current_numpy_state = np.random.RandomState()
            current_numpy_state.set_state(np_state)
            state["random_states"] = (current_numpy_state, current_torch_state)
        self.__dict__ = state

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU')."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)

    def save(self, path):
        """Save the model in the passed `path`."""
        device_backup = self._device
        self.set_device(torch.device("cpu"))
        with open(path, "wb") as output:
            cloudpickle.dump(self, output)
        self.set_device(device_backup)

    @classmethod
    def load(cls, path, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Load the model stored in the passed `path`."""
        with open(path, "rb") as f:
            model = cloudpickle.load(f)
        model.set_device(device)
        return model

    def set_random_state(self, random_state):
        """Set the random state.

        Args:
            random_state (int, tuple, or None):
                Either a tuple containing the (numpy.random.RandomState, torch.Generator)
                or an int representing the random seed to use for both random states.
        """
        if random_state is None:
            self.random_states = random_state
        elif isinstance(random_state, int):
            self.random_states = (
                np.random.RandomState(seed=random_state),
                torch.Generator().manual_seed(random_state),
            )
        elif (
            isinstance(random_state, tuple)
            and isinstance(random_state[0], np.random.RandomState)
            and isinstance(random_state[1], torch.Generator)
        ):
            self.random_states = random_state
        else:
            raise TypeError(
                f"`random_state` {random_state} expected to be an int or a tuple of "
                "(`np.random.RandomState`, `torch.Generator`)"
            )
