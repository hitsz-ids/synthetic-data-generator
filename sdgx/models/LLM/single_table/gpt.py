from __future__ import annotations

import os
import random
import re
from copy import copy

import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.exceptions import InitializationError
from sdgx.models.LLM.base import LLMBaseModel
from sdgx.utils import logger


class SingleTableGPTModel(LLMBaseModel):
    """
    This is a synthetic data generation model powered by OpenAI GPT, a state-of-the-art language model. This model is based on groundbreaking research presented in the ICLR paper titled "Language Models are Realistic Tabular Data Generators".

    Our model harnesses the power of GPT to generate synthetic tabular data that closely resembles real-world datasets. By utilizing the advanced capabilities of GPT, we aim to provide a reliable and efficient solution for generating simulated data that can be used for various purposes, such as testing, training, and analysis.

    With this synthetic data generation model, users can easily generate diverse and realistic tabular datasets, mimicking the characteristics and patterns found in real data.
    """

    openai_API_key = ""
    """
    The API key required to access the OpenAI GPT model. Please provide your own API key for authentication.
    """

    openai_API_url = "https://api.openai.com/v1/"
    """
    The URL endpoint for the OpenAI GPT API. Please specify the appropriate URL for accessing the API.
    """

    max_tokens = 4000
    """
    The maximum number of tokens allowed in the generated response. This parameter helps in limiting the length of the output text.
    """

    temperature = 0.1
    """
    A parameter that controls the randomness of the generated text. Lower values like 0.1 make the output more focused and deterministic, while higher values like 1.0 introduce more randomness.
    """

    timeout = 90
    """
    The maximum time (in seconds) to wait for a response from the OpenAI GPT API. If the response is not received within this time, the request will be timed out.
    """

    gpt_model = "gpt-3.5-turbo"
    """
    The specific GPT model to be used for generating text. The default model is "gpt-3.5-turbo", which is known for its high performance and versatility.
    """

    query_batch = 30
    """
    This parameter is the number of samples submitted to GPT each time and the number of returned samples.

    This size has a certain relationship with the max_token parameter.

    We do not recommend setting too large a value, as this may cause potential problems or errors.
    """

    _sample_lines = []
    """
    A list to store the sample lines of generated data.
    """

    _result_list = []
    """
    A list to store the generated data samples.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the class instance.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self._get_openai_setting_from_env()

    def check(self):
        """
        Performs various checks.

        Raises:
            SynthesizerInitError: If data access type is not specified or if duplicate data access type is found.
        """
        self._check_openAI_setting()
        self._set_openAI()
        self._check_access_type()

    def set_openAI_settings(self, API_url="https://api.openai.com/v1/", API_key=""):
        """
        Sets the OpenAI settings.

        Args:
            API_url (str): The OpenAI API URL. Defaults to "https://api.openai.com/v1/".
            API_key (str): The OpenAI API key. Defaults to an empty string.
        """
        self.openai_API_url = API_url
        self.openai_API_key = API_key
        self._set_openAI()

    def _set_openAI(self):
        """
        Sets the OpenAI API key and base URL.
        """
        openai.api_key = self.openai_API_key
        openai.base_url = self.openai_API_url

    def _check_openAI_setting(self):
        """
        Checks if the OpenAI settings are properly initialized.

        Raises:
            InitializationError: If openai_API_url or openai_API_key is not found.
        """
        if not self.openai_API_url:
            raise InitializationError("openai_API_url NOT found.")
        if not self.openai_API_key:
            raise InitializationError("openai_API_key NOT found.")
        logger.debug("OpenAI setting check passed.")

    def _get_openai_setting_from_env(self):
        """
        Retrieves OpenAI settings from environment variables.
        """
        if os.getenv("OPENAI_KEY"):
            self.openai_API_key = os.getenv("OPENAI_KEY")
            logger.debug("Get OPENAI_KEY from ENV.")
        if os.getenv("OPENAI_URL"):
            self.openai_API_url = os.getenv("OPENAI_URL")
            logger.debug("Get OPENAI_URL from ENV.")

    def openai_client(self):
        """
        Generate a openai request client.
        """
        return openai.OpenAI(api_key=self.openai_API_key, base_url=self.openai_API_url)

    def ask_gpt(self, question, model=None):
        """
        Sends a question to the GPT model.

        Args:
            question (str): The question to ask.
            model (str): The GPT model to use. Defaults to None.

        Returns:
            str: The response from the GPT model.

        Raises:
            SynthesizerInitError: If the check method fails.
        """
        self.check()
        if model:
            model = model
        else:
            model = self.gpt_model
        client = self.openai_client()
        logger.info(f"Ask GPT with temperature = {self.temperature}.")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": question,
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        logger.info("Ask GPT Finished.")
        # store response
        self._responses.append(response)
        # return the content of the gpt response
        return response.choices[0].message.content

    def fit(
        self, raw_data: pd.DataFrame | DataLoader = None, metadata: Metadata = None, *args, **kwargs
    ):
        """
        Fits this model to the provided data.
        Please note that no actual algorithmic training is excuted here.

        Args:
            raw_data (pd.DataFrame | DataLoader): The raw data to fit the model to. It can be either a pandas DataFrame or a DataLoader object.
            metadata (Metadata): The metadata associated with the raw data.

        Returns:
            None

        Raises:
            InitializationError: If neither raw_data nor metadata is provided.
        """

        if raw_data is not None and type(raw_data) in [pd.DataFrame, DataLoader]:
            if metadata:
                self._metadata = metadata
            self._fit_with_data(raw_data)
            return

        if type(raw_data) is Metadata:
            self._fit_with_metadata(raw_data)
            return

        if metadata is not None and type(metadata) is Metadata:
            self._fit_with_metadata(metadata)
            return

        raise InitializationError(
            "Ple1ase pass at least one valid parameter, train_data or metadata"
        )

    def _fit_with_metadata(self, metadata):
        """
        Fit the model using metadata.

        Args:
            metadata: Metadata object.

        Returns:
            None
        """
        logger.info("Fitting model with metadata...")
        self.use_metadata = True
        self._metadata = metadata
        self.columns = list(metadata.column_list)
        logger.info("Fitting model with metadata... Finished.")

    def _fit_with_data(self, train_data):
        """
        Fit the model using data.

        Args:
            train_data: Training data.

        Returns:
            None
        """
        logger.info("Fitting model with raw data...")
        self.use_raw_data = True
        self.use_dataloader = False
        if type(train_data) is DataLoader:
            self.columns = list(train_data.columns())
            train_data = train_data.load_all()
        if not self.columns:
            self.columns = list(train_data.columns)
        # get metadata if no metadata
        if not self._metadata:
            self._metadata = Metadata.from_dataframe(train_data)
        # here we got both raw_data and metadata
        sample_lines = []
        for _, row in train_data.iterrows():
            each_line = ""
            shuffled_columns = copy(self.columns)
            random.shuffle(shuffled_columns)
            for column in shuffled_columns:
                value = str(row[column])
                each_line += f"{column} is {value}, "

            each_line = each_line[:-2]
            each_line += "\n"
            sample_lines.append(each_line)
        self._sample_lines = sample_lines
        logger.info("Fitting model with raw data... Finished.")

    @staticmethod
    def _select_random_elements(input_list, cnt):
        """
        This function selects a random sample of elements from the input list.

        Args:
            input_list (list): The list from which elements will be selected.
            cnt (int): The number of elements to be selected.

        Returns:
            list: A list of randomly selected elements from the input list.

        Raises:
            ValueError: If cnt is greater than the length of the input list.
        """
        if cnt > len(input_list):
            raise ValueError("cnt should not be greater than the length of the list")
        return random.sample(input_list, cnt)

    def _form_message_with_data(self, sample_list, current_cnt):
        """
        This function forms a message with data.

        Args:
            sample_list (list): A list of samples.
            current_cnt (int): The current count of samples.

        Returns:
            str: The formed message with data.
        """
        # form sample string
        sample_str = ""
        for i in range(current_cnt):
            each_sample = sample_list[i]
            each_str = f"sample {i}: " + each_sample + "\n"
            sample_str += each_str

        # form the message sent to GPT
        message = self.prompts["message_prefix"] + sample_str

        # add dataset description
        message = message + self._form_dataset_description()

        # add off-table features
        message = message + self._form_message_with_offtable_features()

        message = (
            message
            + f"Please note that the generated table has total {len(self.columns) + len(self.off_table_features)} columns of the generated data, the column names are {self.columns + self.off_table_features}, every column should not be missed when generating the data. \n"
        )

        # add the suffix of the message
        message = message + self.prompts["message_suffix"] + str(current_cnt) + "."
        # Maybe it can be optimized here
        self._message_list.append(message)
        logger.debug("Message Generated.")
        return message

    def extract_samples_from_response(self, response_content):
        """
        Extracts samples from the response content.

        Args:
            response_content (dict): The response content as a dictionary.

        Returns:
            list: A list of extracted samples.
        """

        def dict_to_list(input_dict, header):
            """
            Converts a dictionary to a list based on the given header.

            Args:
                input_dict (dict): The input dictionary.
                header (list): The list of keys to extract from the dictionary.

            Returns:
                list: A list of values extracted from the dictionary based on the header.
            """
            res = []
            for each_col in header:
                each_value = input_dict.get(each_col, None)
                res.append(each_value)
            return res

        logger.info("Extracting samples from response ...")
        header = self.columns + self.off_table_features
        features = []
        for line in response_content.split("\n"):
            feature = {}
            for field in header:
                pattern = r"\b" + field + r"\s*(?:is|=)\s*([^,\n]+)"
                match = re.search(pattern, line)
                if match:
                    feature[field] = match.group(1).strip()
            if feature:
                features.append(dict_to_list(feature, header))
        logger.info(f"Extracting samples from response ... Finished, {len(features)} extracted.")
        return features

    def sample(self, count=50, dataset_desp="", *args, **kwargs):
        """
        This function samples data from either raw data or metadata based on the given parameters.

        Args:
            count (int): The number of samples to be generated. Default is 50.
            dataset_desp (str): The description of the dataset. Default is an empty string.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            res: The sampled data.
        """
        logger.info("Sampling use GPT model ...")
        self.dataset_description = dataset_desp

        if self.use_raw_data:
            # If the use_raw_data flag is True, sample data using the _sample_with_data method.
            res = self._sample_with_data(count, *args, **kwargs)

        elif self.use_metadata:
            # If the use_metadata flag is True, sample data using the _sample_with_metadata method.
            res = self._sample_with_metadata(count, *args, **kwargs)
        logger.info("Sampling use GPT model ... Finished.")
        return res

    def _form_message_with_metadata(self, current_cnt):
        """
        This function forms a message with metadata for table data generation task.

        Args:
            current_cnt (int): The current count of the message.

        Returns:
            str: The formed message with metadata.
        """
        # form message
        message = ""
        message = message + self.prompts["message_prefix"]  # Add the message prefix
        message = message + self._form_dataset_description()  # Add the dataset description
        message = (
            message
            + "This table data generation task will only have metadata and no data samples. The header (columns infomation) of the tabular data is: "
        )
        # Add information about the table data generation task
        message = message + str(self.columns) + ". \n"  # Add the column information
        message = (
            message + self._form_message_with_offtable_features()
        )  # Add off-table features information
        message = (
            message
            + f"Note that the generated table has total {len(self.columns) + len(self.off_table_features)} columns, the column names are {self.columns + self.off_table_features}, every column should NOT be missed in generated data.\n"
        )  # Add information about the generated table columns

        # Add the message suffix and current count
        message = message + self.prompts["message_suffix"] + str(current_cnt) + "."
        # Append the message to the message list
        self._message_list.append(message)
        # Return the formed message with metadata
        return message

    def _sample_with_metadata(self, count, *args, **kwargs):
        """
        This method samples data with metadata.

        Args:
            count (int): The number of samples to be generated.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            int: The input count.

        """
        logger.info("Sampling with metadata.")
        # Initialize an empty list to store the generated samples
        result = []
        # Set the remaining count to the input count
        remaining_cnt = count

        # Set the remaining count to the input count
        while remaining_cnt > 0:
            if remaining_cnt - self.query_batch >= 0:
                # Set the current count to the query batch size if >= 0
                current_cnt = self.query_batch
            else:
                # else, set the current count to the remaining count
                current_cnt = remaining_cnt
            # Generate a message with metadata
            message = self._form_message_with_metadata(current_cnt)
            # Send the message to GPT and get the response
            response = self.ask_gpt(message)
            # Extract features from the response
            generated_batch = self.extract_samples_from_response(response)
            # Add the generated batch to the result list
            result += generated_batch
            # Update the remaining count
            remaining_cnt = remaining_cnt - current_cnt

        self._result_list.append(result)

        # return result in pd.DataFrame
        final_columns = self.columns + self.off_table_features
        return pd.DataFrame(result, columns=final_columns)

    def _sample_with_data(self, count, *args, **kwargs):
        """
        This function samples data with a given count and returns a DataFrame with the sampled data.

        Args:
            count (int): The number of data samples to be generated.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pd.DataFrame: A DataFrame containing the sampled data.

        """
        logger.info("Sampling with raw_data.")
        result = []
        remaining_cnt = count
        while remaining_cnt > 0:
            # get the current count
            if remaining_cnt - self.query_batch >= 0:
                current_cnt = self.query_batch
            else:
                current_cnt = remaining_cnt
            # select data / form message
            sample_list = self._select_random_elements(self._sample_lines, current_cnt)
            message = self._form_message_with_data(sample_list, current_cnt)
            # ask_gpt
            response = self.ask_gpt(message)
            # get result from response
            generated_batch = self.extract_samples_from_response(response)
            # update result
            result += generated_batch
            # update remaining_cnt
            remaining_cnt = remaining_cnt - current_cnt

        self._result_list.append(result)

        # return result
        final_columns = self.columns + self.off_table_features
        return pd.DataFrame(result, columns=final_columns)
