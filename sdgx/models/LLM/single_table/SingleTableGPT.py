from __future__ import annotations

from pathlib import Path
import pandas as pd
import openai
import os
import random
import re 

from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.exceptions import SynthesizerInitError
from sdgx.models.base import SynthesizerModel
from sdgx.exceptions import InitializationError 


class SingleTableGPTModel(SynthesizerModel):
    """
    This is a synthetic data generation model powered by openAI GPT - a state-of-the-art language model. This model is based on the groundbreaking research presented in the ICLR paper titled "Language Models are Realistic Tabular Data Generators".

    Our model leverages the power of GPT to generate synthetic tabular data that closely resembles real-world datasets. By utilizing the advanced capabilities of GPT, we aim to provide a reliable and efficient solution for generating simulated data that can be used for various purposes, such as testing, training, and analysis.
    
    With this synthetic data generation model, one can easily generate diverse and realistic tabular datasets, mimicking the characteristics and patterns found in real data. 
    """

    fit_medatada = False
    fit_raw_data = False

    openai_API_key = ""
    """
    The API key required to access the OpenAI GPT model. Please provide your own API key for authentication.
    """

    openai_API_url = "https://api.openai.com/v1/"
    """
    The URL endpoint for the OpenAI GPT API. Please specify the appropriate URL for accessing the API.
    """

    max_tokens=2000
    """
    The maximum number of tokens allowed in the generated response. This parameter helps in limiting the length of the output text.
    """

    temperature=0.1
    """A parameter that controls the randomness of the generated text. Lower values like 0.1 make the output more focused and deterministic, while higher values like 1.0 introduce more randomness.
    """

    timeout=60
    """
    The maximum time (in seconds) to wait for a response from the OpenAI GPT API. If the response is not received within this time, the request will be timed out.
    """

    gpt_model = "Gpt-3.5-turbo-0613"
    """
    The specific GPT model to be used for generating text. The default model is "gpt-3.5-turbo", which is known for its high performance and versatility.
    """
    
    use_raw_data = False
    ''' 
    By default, we use raw_data for data access. 

    When using the data loader, due to the randomization operation involved, we currently use the `.load_all()` operation to take out the original data in pd.DataFrame format for subsequent processing.

    Due to the characteristics of the OpenAI GPT service, we do not recommend running this model with large data tables, which will consume your tokens excessively.
    '''

    query_batch = 20
    '''
    This parameter is the number of samples submitted to GPT each time and the number of returned samples. 
    
    This size has a certain relationship with the max_token parameter.

    We do not recommend setting too large a value, as this may cause potential problems or errors.
    '''

    off_table_feature_inference = []
    '''
    * Experimental Feature
    
    Whether infer data columns that do not exist in the real data table, the effect may not be very good.
    '''
    
    prompts = {
        "message_prefix" : '''Suppose you are the best data generating model in this world, we have some data entries with the following information:\n''',
        "message_suffix" :"""\nGenerate synthetic data samples based on the above information, the count of the generated data samples is""",
        "system_role_content":"You are a powerful synthetic data generation model."
    }
    """
    Prompt words for generating data (preliminary version, improvements welcome).
    """

    _sample_lines = []

    _responses = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def check(self):

        self._check_access_type()
        self._check_openAI_setting()
        self._set_openAI_key()

    def set_openAI_settings(self, API_url, API_key):
        self.openai_API_url = API_url
        self.openai_API_key = API_key

        self._set_openAI_key()

    def _set_openAI_key(self):
        openai.api_key = self.openai_API_key
        openai.api_base = self.openai_API_url
    
    def _check_openAI_setting(self):
        if not self.openai_API_url:
            raise InitializationError('openai_API_url NOT found.')
        if not self.openai_API_key:
            raise InitializationError('openai_API_key NOT found.')
        pass

    def _get_openai_setting_from_env(self):
        if os.getenv("OPENAI_KEY"):
            self.openai_API_key = os.getenv("OPENAI_KEY")
        if os.getenv("OPENAI_URL"):
            self.openai_API_url = os.getenv("OPENAI_URL")

    def ask_gpt(self, question, model="gpt-3.5-turbo"):
        api_key = self.openai_API_key
        if model:
            model = model
        else:
            model = self.gpt_model
        openai.api_key = api_key
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": question,
                },
            ],
            temperature= self.temperature, 
            max_tokens= self.max_tokens,
            timeout=self.timeout
        )
        # store response 
        self._responses.append(response)
        # return response
        return response.choices[0].message.content

    def fit(self, train_data: pd.DataFrame | DataLoader = None, metadata: Metadata = None, *args, **kwargs):

        if train_data:
            self._fit_with_data(train_data)
        elif metadata:
            self._fit_with_metadata(metadata)
        else:
            raise InitializationError('Please pass at least one valid parameter, train_data or metadata')
    
    def _fit_with_metadata(self, metadata):
        self._metadata = metadata
        self.columns  = metadata.column_list
        pass
    
    def _fit_with_data(self, train_data):
        if type(train_data) is DataLoader:
            train_data = train_data.load_all()
        sample_lines = []
        col_list = list(train_data.columns)
        self.columns = col_list
        
        for index, row in train_data.iterrows():
            each_line = ''
            random.shuffle(col_list)
            for column in col_list:
                value = str(row[column])
                each_line += f"{column} is {value}, "
            
            each_line = each_line[:-2]
            each_line += "\n"
            sample_lines.append(each_line)
        self._sample_lines = sample_lines
    
    @staticmethod
    def _select_random_elements(input_list, cnt):
        if cnt > len(input_list):
            raise ValueError("cnt should not be greater than the length of the list")
        return random.sample(input_list, cnt)
    
    def _split_data_entries(data):
        lines = data.split("sample")
        lines = [line.strip() for line in lines if line.strip()]
        # remove the first line 
        return lines[1:] 

    def _extract_features(self, sample_string):
        original_columns = self.columns
        features = []
        # Maybe it can be optimized here
        pattern = r'(\w+)\sis\s(\w+|\?)'
        matches = re.findall(pattern, sample_string)
        for column in original_columns:
            for match in matches:
                if match[0] == column:
                    features.append(match[1])
                    break
            else:
                features.append(None)
        return features

    def _form_message(self, sample_list, current_cnt):
        # form sample string 
        sample_str = ""
        for i in range(current_cnt):
            each_sample = sample_list[i]
            each_str = f"sample {i}: " + each_sample + "\n"
            sample_str += each_str
        # form the message sent to GPT 
        message = self.prompts['message_prefix'] + sample_str + f'\nPlease note that the table has a total of {len(self.columns)} columns of data, which should not be missing when generating the data.  '+ self.prompts["message_suffix"] + str(current_cnt) + '.'

        return message
    
    def _get_result_from_response(self, response: str):
        result = []
        response_list = self._split_data_entries(response)
        for each_response in response_list:
            each_data = self._extract_features(each_response)
            result.append(each_data)
        return result


    def sample(self, count = 50, *args, **kwargs):

        if self._fit_with_data:
            res = self._sample_with_data(count,  *args, **kwargs)
            
        elif self._fit_with_metadata:
            res = self._sample_with_metadata(count, *args, **kwargs)
        return res 
    
    def _sample_with_medadata(self, count, *args, **kwargs):
        # not implemented

        pass
    
    def _sample_with_data(self, count, *args, **kwargs):
        result = []
        remaining_cnt = count
        while remaining_cnt > 0:
            # get the current count 
            if remaining_cnt - self.query_batch >=0:
                current_cnt = self.query_batch
            else:
                current_cnt = remaining_cnt
            # select data / form message
            sample_list = self._select_random_elements(self._sample_lines, current_cnt)
            message = self._form_message(sample_list, current_cnt)
            # ask_gpt
            response = self.ask_gpt(message)
            # get result from response 
            generated_batch = self._get_result_from_response(response)
            # update result 
            result += generated_batch
            # update remaining_cnt 
            remaining_cnt = remaining_cnt - current_cnt
        
        # return pd DataFrame 
        return pd.DataFrame(result, columns = self.columns, index = False)


