from sdgx.exceptions import SynthesizerInitError
from sdgx.models.base import SynthesizerModel
from sdgx.utils import logger


class LLMBaseModel(SynthesizerModel):
    """
    This is a base class for generating synthetic data using LLM (Large Language Model).

    Note:
    - When using the data loader, the original data is transformed to pd.DataFrame format for subsequent processing.
    - It is not recommended to use this model with large data tables due to excessive token consumption in some expensive LLM service.
    - Generating data based on metadata is a potential way to generate data that cannot be made public and contains sensitive information.
    """

    use_raw_data = False
    """
    By default, we use raw_data for data access.

    When using the data loader, due to the need of randomization operation, we currently use the `.load_all()` to transform the original data to pd.DataFrame format for subsequent processing.

    Due to the characteristics of the OpenAI GPT service, we do not recommend running this model with large data tables, which will consume your tokens excessively.
    """

    use_metadata = False
    """
    In this model, we accept a data generation paradigm that only provides metadata.

    When only metadata is provided, sdgx will format the metadata of the data set into a message and transmit it to GPT, and GPT will generate similar data based on what it knows.

    This is a potential way to generate data that cannot be made public and contains sensitive information.
    """

    _metadata = None
    """
    the metadata.
    """

    off_table_features = []
    """
    * Experimental Feature

    Whether infer data columns that do not exist in the real data table, the effect may not be very good.
    """

    prompts = {
        "message_prefix": """Suppose you are the best data generating model in this world, we have some data samples with the following information:\n\n""",
        "message_suffix": """\nGenerate synthetic data samples based on the above information and your knowledge, each sample should be output on one line (do not output in multiple lines), the output format of the sample is the same as the example in this message, such as "column_name_1 is value_1", the count of the generated data samples is """,
        "system_role_content": "You are a powerful synthetic data generation model.",
    }
    """
    Prompt words for generating data (preliminary version, improvements welcome).
    """

    columns = []
    """
    The columns of the data set.
    """

    dataset_description = ""
    """
    The description of the data set.
    """

    _responses = []
    """
    A list to store the responses received from the LLM.
    """

    _message_list = []
    """
    A list to store the messages used to ask LLM.
    """

    def _check_access_type(self):
        """
        Checks the data access type.

        Raises:
            SynthesizerInitError: If data access type is not specified or if duplicate data access type is found.
        """
        if self.use_dataloader == self.use_raw_data == self.use_metadata == False:
            raise SynthesizerInitError(
                "Data access type not specified, please use `use_raw_data: bool` or `use_dataloader: bool` to specify data access type."
            )
        if self.use_dataloader == self.use_raw_data == True:
            raise SynthesizerInitError("Duplicate data access type found.")

    def _form_columns_description(self):
        """
        We believe that giving information about a column helps improve data quality.

        Currently, we leave this function to Good First Issue until March 2024, if unclaimed we will implement it quickly.
        """

        raise NotImplementedError

    def _form_message_with_offtable_features(self):
        """
        This function forms a message with off-table features.

        If there are more off-table columns, additional processing is excuted here.
        """
        if self.off_table_features:
            logger.info(f"Use off_table_feature = {self.off_table_features}.")
            return f"Also, you should try to infer another {len(self.off_table_features)} columns based on your knowledge, the name of these columns are : {self.off_table_features}, attach these columns after the original table. \n"
        else:
            logger.info("No off_table_feature needed in current model.")
            return ""

    def _form_dataset_description(self):
        """
        This function is used to form the dataset description.

        Returns:
            str: The description of the generated table.
        """
        if self.dataset_description:
            logger.info(f"Use dataset_description = {self.dataset_description}.")
            return "\nThe description of the generated table is " + self.dataset_description + "\n"
        else:
            logger.info("No dataset_description given in current model.")
            return ""
