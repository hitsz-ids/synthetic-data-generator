

class columnMetric():

    upper_bound = None
    lower_bound = None
    metric_name = None

    def __init__(self) -> None:
        
        pass
    
    @classmethod
    def check_input(real_data, synthetic_data):
        # should be list \ Series \ 1d-Array

        pass

    @classmethod
    def calculate(real_data, synthetic_data):

        raise NotImplementedError()

    @classmethod
    def check_output(raw_metric_value):

        raise NotImplementedError()

    pass