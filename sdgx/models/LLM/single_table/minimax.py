from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from sdgx.exceptions import InitializationError
from sdgx.models.LLM.single_table.gpt import SingleTableGPTModel
from sdgx.utils import logger

# MiniMax supported models
MINIMAX_MODELS = ["MiniMax-M2.7", "MiniMax-M2.7-highspeed"]


class SingleTableMiniMaxModel(SingleTableGPTModel):
    """
    Synthetic data generation model powered by MiniMax LLM.

    MiniMax provides OpenAI-compatible API, so this model inherits from
    SingleTableGPTModel and overrides the configuration to use MiniMax's
    API endpoint and models.

    Supported models:
    - MiniMax-M2.7: Peak Performance. Ultimate Value. Master the Complex.
    - MiniMax-M2.7-highspeed: Same performance, faster and more agile.

    Environment variables:
    - MINIMAX_API_KEY: API key for MiniMax (required)
    - MINIMAX_API_URL: Custom base URL (optional, defaults to https://api.minimax.io/v1)

    Usage::

        from sdgx.models.LLM.single_table.minimax import SingleTableMiniMaxModel

        model = SingleTableMiniMaxModel()
        model.fit(raw_data)
        synthetic_data = model.sample(100)

    For more information, see: https://platform.minimax.io/docs/api-reference/text-openai-api
    """

    openai_API_url = "https://api.minimax.io/v1"
    """
    The base URL for MiniMax's OpenAI-compatible API.
    """

    gpt_model = "MiniMax-M2.7"
    """
    The default MiniMax model. Use 'MiniMax-M2.7-highspeed' for faster responses.
    """

    temperature = 1.0
    """
    MiniMax temperature must be in (0.0, 1.0]. Cannot be 0. Default is 1.0.
    """

    def __init__(self, *args, **kwargs) -> None:
        if "temperature" in kwargs:
            self.temperature = kwargs.pop("temperature")
        super().__init__(*args, **kwargs)
        self._get_minimax_setting_from_env()
        self._clamp_temperature()

    def _get_minimax_setting_from_env(self):
        """
        Retrieves MiniMax settings from environment variables.
        Falls back to OPENAI_KEY/OPENAI_URL if MINIMAX_* vars are not set.
        """
        if os.getenv("MINIMAX_API_KEY"):
            self.openai_API_key = os.getenv("MINIMAX_API_KEY")
            logger.debug("Get MINIMAX_API_KEY from ENV.")
        if os.getenv("MINIMAX_API_URL"):
            self.openai_API_url = os.getenv("MINIMAX_API_URL")
            logger.debug("Get MINIMAX_API_URL from ENV.")

    def _clamp_temperature(self):
        """
        MiniMax requires temperature in (0.0, 1.0].
        Clamp values outside this range.
        """
        if self.temperature <= 0:
            self.temperature = 0.01
            logger.warning("MiniMax does not support temperature=0, clamped to 0.01.")
        elif self.temperature > 1.0:
            self.temperature = 1.0
            logger.warning("MiniMax temperature capped at 1.0.")

    def _check_openAI_setting(self):
        """
        Checks if the MiniMax API settings are properly initialized.
        """
        if not self.openai_API_url:
            raise InitializationError("MiniMax API URL not found.")
        if not self.openai_API_key:
            raise InitializationError(
                "MiniMax API key not found. "
                "Set MINIMAX_API_KEY environment variable or call "
                "set_minimax_settings(API_key='your-key')."
            )
        logger.debug("MiniMax setting check passed.")

    def set_minimax_settings(self, API_url="https://api.minimax.io/v1", API_key=""):
        """
        Sets the MiniMax API settings.

        Args:
            API_url (str): The MiniMax API URL. Defaults to "https://api.minimax.io/v1".
            API_key (str): The MiniMax API key.
        """
        self.openai_API_url = API_url
        self.openai_API_key = API_key
        self._set_openAI()

    def ask_gpt(self, question, model=None):
        """
        Sends a question to the MiniMax model via OpenAI-compatible API.

        Ensures temperature is within MiniMax's valid range before making the call.
        """
        self._clamp_temperature()
        return super().ask_gpt(question, model=model)
