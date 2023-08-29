from typing import List, Optional

import numpy as np
import torch


class BaseSynthesizerModel:
    random_states = None

    def __init__(self, transformer=None, sampler=None) -> None:
        # 以下几个变量都需要在初始化 model 时进行更改
        self.model = None  # 存放模型
        self.status = "UNFINED"
        self.model_type = "MODEL_TYPE_UNDEFINED"
        # self.epochs = epochs
        self._device = "CPU"

    def fit(self, input_df, discrete_cols: Optional[List] = None):
        raise NotImplementedError

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU')."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)

    def __getstate__(self):
        device_backup = self._device
        self.set_device(torch.device("cpu"))
        state = self.__dict__.copy()
        self.set_device(device_backup)
        if (
            isinstance(self.random_states, tuple)
            and isinstance(self.random_states[0], np.random.RandomState)
            and isinstance(self.random_states[1], torch.Generator)
        ):
            state["_numpy_random_state"] = self.random_states[0].get_state()
            state["_torch_random_state"] = self.random_states[1].get_state()
            state.pop("random_states")
        return state

    def __setstate__(self, state):
        if "_numpy_random_state" in state and "_torch_random_state" in state:
            np_state = state.pop("_numpy_random_state")
            torch_state = state.pop("_torch_random_state")
            current_torch_state = torch.Generator()
            current_torch_state.set_state(torch_state)
            current_numpy_state = np.random.RandomState()
            current_numpy_state.set_state(np_state)
            state["random_states"] = (current_numpy_state, current_torch_state)
        self.__dict__ = state
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.set_device(device)

    def save(self, path):
        device_backup = self._device
        self.set_device(torch.device("cpu"))
        torch.save(self, path)
        self.set_device(device_backup)

    @classmethod
    def load(cls, path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.load(path)
        model.set_device(device)
        return model

    def set_random_state(self, random_state):
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
