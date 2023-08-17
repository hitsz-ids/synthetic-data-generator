from abc import ABC, abstractmethod


class BaseMetric(ABC):
    def __init__(self, real_data, synthetic_data):
        self.real_data = real_data
        self.synthetic_data = synthetic_data

    @abstractmethod
    def calculate(self):
        pass

    def validate_datasets(self):
        pass
