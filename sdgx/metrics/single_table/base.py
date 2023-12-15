class singleTableMetric:
    upper_bound = None
    lower_bound = None
    metric_name = None
    metadata = None

    def __init__(self, metadata) -> None:
        self.metadata = metadata
        pass

    @classmethod
    def check_input(real_data, synthetic_data):
        # should be pd.DataFrame

        raise NotImplementedError()

    # not a class method
    def calculate(real_data, synthetic_data):
        raise NotImplementedError()

    @classmethod
    def check_output(raw_metric_value):
        raise NotImplementedError()

    pass
