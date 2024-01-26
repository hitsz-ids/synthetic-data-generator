from __future__ import annotations

from pathlib import Path
import pandas as pd
import openai
import os
import random

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

    openai_API_key = ""
    """
    The API key required to access the OpenAI GPT model. Please provide your own API key for authentication.
    """

    openai_API_url = ""
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

    gpt_model = "gpt-3.5-turbo"
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
        "message_suffix" :"""Generate synthetic data samples based on the above information, the count of the generated data samples is""",
        "system_role_content":"You are a powerful synthetic data generation model."
    }
    """
    Prompt words for generating data (preliminary version, improvements welcome).
    """

    _sample_lines = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def check(self):

        self._check_access_type()
        self._check_LLM_setting()
        
        pass
    
    def _check_LLM_setting(self):
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

    def _count_tokens(self, question):
        
        response = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": self.prompts['system_role_content']},
                {"role": "user", "content": question}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )

        return response['usage']['total_tokens']

    def ask_gpt(self, question):
        # 设置聊天的起始对话
        start_chat = 'User: ' + question + '\nAI:'

        # 调用 OpenAI 的 ChatCompletion API
        response = openai.ChatCompletion.create(
             model=self.gpt_model,
            messages=[
                {"role": "system", "content": self.prompts['system_role_content']},
                {"role": "user", "content": start_chat}
            ],
            max_tokens=self.max_tokens, 
            temperature=self.temperature,
            timeout=self.timeout
        )

        return response.choices[0].message.content

    def fit(self, train_data: pd.DataFrame | DataLoader ):
        if type(train_data) is DataLoader:
            train_data = train_data.load_all()
        
        sample_lines = []
        col_list = list(train_data.columns)
        # 遍历每一行数据
        for index, row in train_data.iterrows():
            # 遍历每一列数据
            each_line = ''
            random.shuffle(col_list)
            for column in col_list:
                # 将列名和当前行的数值拼接为字符串
                value = str(row[column])
                each_line += f"{column} is {value}, "
            
            # 移除最后一个逗号和空格
            each_line = each_line[:-2]
            # 添加换行符
            each_line += "\n"
            sample_lines.append(each_line)
        self._sample_lines = sample_lines

    def sample(self, count):

        pass


