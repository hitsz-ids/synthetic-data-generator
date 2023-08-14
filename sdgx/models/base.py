import numpy as np
import torch


class BaseGeneratorModel:
    def __init__(self, transformer=None, sampler=None) -> None:
        # 以下几个变量都需要在初始化 model 时进行更改
        self.model = None  # 存放模型
        self.status = "UNFINED"
        self.model_type = "MODEL_TYPE_UNDEFINED"
        # self.epochs = epochs

        # 目前使用CPU计算，后续扩展使用 GPU 及其 Proxy
        self.device = "CPU"

    # fit 模型
    def fit(self):
        # 需要覆写该方法
        raise NotImplementedError

    def generate(self, n_rows=100):
        # 需要覆写该方法
        raise NotImplementedError

    def fit(self):
        # 需要覆写该方法
        raise NotImplementedError

    def load_from_disk(self, model_path=""):
        # 需要覆写该方法
        raise NotImplementedError

    def dump_to_disk(self, output_path=""):
        raise NotImplementedError

    pass
