from __future__ import annotations

import os
import random
import re
from copy import copy

import pandas as pd

from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.exceptions import InitializationError
from sdgx.models.LLM.base import LLMBaseModel
from sdgx.utils import logger


class SingleTableLiteLLMModel(LLMBaseModel):
    """
    Synthetic data generation model powered by LiteLLM, providing access to
    100+ LLM providers (OpenAI, Anthropic, Google, Groq, Together AI, AWS
    Bedrock, Azure, Mistral, etc.) through a single unified interface.

    Uses provider-prefixed model names, e.g. ``openai/gpt-4o``,
    ``anthropic/claude-sonnet-4-6``, ``groq/llama-3.3-70b-versatile``.
    API keys are read from environment variables automatically by LiteLLM.
    """

    model = "openai/gpt-4o-mini"
    """LiteLLM model string (provider/model format)."""

    api_key = ""
    """Optional API key. When empty, LiteLLM reads from provider env vars."""

    api_base = ""
    """Optional API base URL (for LiteLLM proxy or custom endpoints)."""

    max_tokens = 4000
    """Maximum number of tokens in the generated response."""

    temperature = 0.1
    """Sampling temperature. Lower = more deterministic."""

    timeout = 90
    """Maximum seconds to wait for a response."""

    query_batch = 30
    """Number of samples per LLM request."""

    _sample_lines = []
    _result_list = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._get_settings_from_env()

    def check(self):
        self._check_access_type()

    def _get_settings_from_env(self):
        if os.getenv("LITELLM_MODEL"):
            self.model = os.getenv("LITELLM_MODEL")
        if os.getenv("LITELLM_API_KEY"):
            self.api_key = os.getenv("LITELLM_API_KEY")
        if os.getenv("LITELLM_API_BASE"):
            self.api_base = os.getenv("LITELLM_API_BASE")

    def ask_llm(self, question, model=None):
        """Send a question to the LLM via LiteLLM.

        Args:
            question: The prompt text.
            model: Override model string. Defaults to self.model.

        Returns:
            The text content of the LLM response.
        """
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm is required for SingleTableLiteLLMModel. "
                "Install with: pip install litellm"
            )

        self.check()

        kwargs = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": question}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "drop_params": True,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        logger.info(f"Ask LiteLLM ({kwargs['model']}) with temperature = {self.temperature}.")
        response = litellm.completion(**kwargs)
        logger.info("Ask LiteLLM Finished.")

        self._responses.append(response)
        return response.choices[0].message.content

    def fit(
        self,
        raw_data: pd.DataFrame | DataLoader = None,
        metadata: Metadata = None,
        *args,
        **kwargs,
    ):
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
            "Please pass at least one valid parameter, train_data or metadata"
        )

    def _fit_with_metadata(self, metadata):
        logger.info("Fitting model with metadata...")
        self.use_metadata = True
        self._metadata = metadata
        self.columns = list(metadata.column_list)
        logger.info("Fitting model with metadata... Finished.")

    def _fit_with_data(self, train_data):
        logger.info("Fitting model with raw data...")
        self.use_raw_data = True
        self.use_dataloader = False
        if type(train_data) is DataLoader:
            self.columns = list(train_data.columns())
            train_data = train_data.load_all()
        if not self.columns:
            self.columns = list(train_data.columns)
        if not self._metadata:
            self._metadata = Metadata.from_dataframe(train_data)
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
        if cnt >= len(input_list):
            return input_list
        return random.sample(input_list, cnt)

    def _form_message_with_rawdata(self, each_cnt):
        selected_lines = self._select_random_elements(self._sample_lines, each_cnt)
        return (
            self.prompts["message_prefix"]
            + "".join(selected_lines)
            + self._form_dataset_description()
            + self._form_message_with_offtable_features()
            + self.prompts["message_suffix"]
            + str(each_cnt)
        )

    def _form_message_with_metadata(self, each_cnt):
        return (
            self.prompts["message_prefix"]
            + str(self._metadata)
            + self._form_dataset_description()
            + self._form_message_with_offtable_features()
            + self.prompts["message_suffix"]
            + str(each_cnt)
        )

    def sample(self, count=50, dataset_desp="", *args, **kwargs):
        if dataset_desp:
            self.dataset_description = dataset_desp
        logger.info(f"Generating {count} samples using LiteLLM...")

        total_asked = 0
        while total_asked < count:
            each_cnt = min(self.query_batch, count - total_asked)
            if self.use_metadata:
                message = self._form_message_with_metadata(each_cnt)
            else:
                message = self._form_message_with_rawdata(each_cnt)

            self._message_list.append(message)
            response = self.ask_llm(message)

            lines = response.strip().split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                row_data = {}
                pairs = re.split(r",\s*(?=\w+\s+is\s+)", line)
                for pair in pairs:
                    match = re.match(r"(.+?)\s+is\s+(.+)", pair.strip())
                    if match:
                        col_name = match.group(1).strip()
                        col_value = match.group(2).strip()
                        row_data[col_name] = col_value
                if row_data:
                    self._result_list.append(row_data)

            total_asked += each_cnt

        result_df = pd.DataFrame(self._result_list[:count])
        logger.info(f"Generated {len(result_df)} samples.")
        return result_df
