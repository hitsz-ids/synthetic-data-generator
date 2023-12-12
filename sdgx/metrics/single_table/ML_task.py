from copy import copy
from sdgx.log import logger
from sdgx.metrics.single_table.base import SingleTableMetric
# The default setting 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

class SupervisedLearningTask(SingleTableMetric):

    '''SupervisedLearningTask
    
    This module is designed to test the difference between generated data and real data in supervised machine learning. High-quality synthetic data can achieve similar performance and prediction results (such as accuracy, recall, etc.) similar to real data, and avoid potential privacy leakage.
    '''

    def __init__(self, metadata, label_col, model = None, metric_list = []) -> None:
        ''' Create a SupervisedLearningTask for single table synthetic data.
        
        Args:
            - metadata (json): 
                Currently, this parameter accepts a json format as input, and will be replaced by a dedicated metadata object in future versions.

            - label_col (str): 
                This parameter is used to indicate which column belongs to the label column (y) in supervised learning. 

            - model (a sklearn compatible model object):
                The parameter should be a model object, such as sklearn.ensemble._forest.RandomForestClassifier. A model should provide the `.fit()` method for model training and the `.predict()` method for model prediction. The parameter of these two methods need to be compatible with scikit-learn parameter rule.
                If this parameter is none, `RandomForestClassifier` is used by default.

            - metric_list (list of sklearn metric compatible function):
                The parameter should be a list, whose element is a sklearn metric compatible function. Each function takes two parameters `y_true` and `y_pred`. SDG supports calculating multiple machine learning task metrics by adding multiple metric functions to the list.
                If this parameter is none, `accuracy_score` is used by default.
        '''
        
        super().__init__(metadata)

        # result_list
        self.result_list_real = []
        self.result_list_synthetic = []

        # The label column
        self.label_col = label_col
        
        # The model 
        if model is not None:
            self.model_real = model
            self.model_synthetic = copy(self.model_real)
        else: 
            # we use default rf parameters
            self.model_real = RandomForestClassifier()
            self.model_synthetic = RandomForestClassifier()
        # check model
        if hasattr(self.model, 'fit') is False or hasattr(self.model, 'predict') is False:
            raise ValueError("Model should provide the `.fit()` and `.predict()` method.")
        
        # check the metric list 
        if metric_list is not list: 
            raise ValueError("metric_list should be list.")
        # the default metric is accuracy_score
        if metric_list is []:
            self.metric_list = [accuracy_score]
        else:
            self.metric_list = metric_list
    
    def calculate(self, real_data, synthetic_data, test_data):
        # super().calculate(synthetic_data)
        # 1. split test data 
        X_test, y_true = self.split_feature_label(test_data)
        # get 2 y_pred 
        y_pred_real = self.single_ML_task(real_data, X_test, model_type="real")
        y_pred_synthetic = self.single_ML_task(synthetic_data, X_test, model_type='synthetic')
        # calculate 2 metric list 
        
        pass

    def calculate_metrtics(self, y_true, y_pred):
        res_list = []


        pass

    def split_feature_label(self, input_data):
        try:
            y = input_data[self.label_col]
        except KeyError as e:
            logger.exception(KeyError(e))
            exit(1)
        X = input_data.drop(self.label_col, axis = 1)
        return X, y

    def single_ML_task(self, data_table, test_data, model_type) -> list:
        # 1. train the ML model 
        X, y = self.split_feature_label(data_table)
        if model_type == "real":
            self.model_real.fit(X, y)
        else:
            self.model_synthetic.fit(X, y)
        # 2. use test data to get y_pred 
        y_pred = self.model.predict(test_data)
        return y_pred

    pass