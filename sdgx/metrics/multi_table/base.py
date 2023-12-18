
class MultiTableMetric:

    upper_bound = None
    lower_bound = None
    metric_name = None
    metadata = None
    table_list = []

    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata
        pass

    @classmethod
    def check_input(real_data, synthetic_data):
        # real_data or synthetic_data should be dict，其中有表

        pass

    # not a class method
    def calculate(self, real_data, synthetic_data):
        raise NotImplementedError()

    @classmethod
    def check_output(raw_metric_value):
        raise NotImplementedError()

    pass
