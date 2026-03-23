from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.exceptions import InitializationError
from sdgx.models.LLM.single_table.minimax import MINIMAX_MODELS, SingleTableMiniMaxModel


@pytest.fixture
def minimax_model():
    yield SingleTableMiniMaxModel()


@pytest.fixture
def raw_data(demo_single_table_path):
    yield pd.read_csv(demo_single_table_path).head(100)


# --- Unit Tests ---


class TestMiniMaxModelDefaults:
    """Test default configuration of SingleTableMiniMaxModel."""

    def test_default_base_url(self, minimax_model: SingleTableMiniMaxModel):
        assert minimax_model.openai_API_url == "https://api.minimax.io/v1"

    def test_default_model(self, minimax_model: SingleTableMiniMaxModel):
        assert minimax_model.gpt_model == "MiniMax-M2.7"

    def test_default_temperature(self, minimax_model: SingleTableMiniMaxModel):
        assert minimax_model.temperature == 1.0

    def test_default_max_tokens(self, minimax_model: SingleTableMiniMaxModel):
        assert minimax_model.max_tokens == 4000

    def test_default_timeout(self, minimax_model: SingleTableMiniMaxModel):
        assert minimax_model.timeout == 90

    def test_supported_models(self):
        assert "MiniMax-M2.7" in MINIMAX_MODELS
        assert "MiniMax-M2.7-highspeed" in MINIMAX_MODELS


class TestMiniMaxModelSettings:
    """Test configuration and settings."""

    def test_set_minimax_settings(self, minimax_model: SingleTableMiniMaxModel):
        api_key = "test-minimax-key"
        api_url = "https://custom.minimax.io/v1"
        minimax_model.set_minimax_settings(API_url=api_url, API_key=api_key)
        client = minimax_model.openai_client()
        assert client.api_key == api_key
        assert str(client.base_url).rstrip("/") == api_url

    def test_env_minimax_api_key(self, minimax_model: SingleTableMiniMaxModel):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-test-key"}):
            model = SingleTableMiniMaxModel()
            assert model.openai_API_key == "env-test-key"

    def test_env_minimax_api_url(self, minimax_model: SingleTableMiniMaxModel):
        with patch.dict(os.environ, {"MINIMAX_API_URL": "https://custom.url/v1"}):
            model = SingleTableMiniMaxModel()
            assert model.openai_API_url == "https://custom.url/v1"

    def test_missing_api_key_raises_error(self, minimax_model: SingleTableMiniMaxModel):
        minimax_model.openai_API_key = ""
        with pytest.raises(InitializationError, match="MiniMax API key not found"):
            minimax_model._check_openAI_setting()

    def test_missing_api_url_raises_error(self, minimax_model: SingleTableMiniMaxModel):
        minimax_model.openai_API_url = ""
        with pytest.raises(InitializationError, match="MiniMax API URL not found"):
            minimax_model._check_openAI_setting()

    def test_valid_settings_pass_check(self, minimax_model: SingleTableMiniMaxModel):
        minimax_model.openai_API_key = "valid-key"
        minimax_model.openai_API_url = "https://api.minimax.io/v1"
        minimax_model._check_openAI_setting()  # Should not raise


class TestMiniMaxTemperatureClamping:
    """Test temperature clamping behavior."""

    def test_temperature_zero_clamped(self):
        model = SingleTableMiniMaxModel(temperature=0)
        assert model.temperature == 0.01

    def test_temperature_negative_clamped(self):
        model = SingleTableMiniMaxModel(temperature=-0.5)
        assert model.temperature == 0.01

    def test_temperature_above_one_clamped(self):
        model = SingleTableMiniMaxModel(temperature=1.5)
        assert model.temperature == 1.0

    def test_temperature_valid_not_changed(self):
        model = SingleTableMiniMaxModel(temperature=0.7)
        assert model.temperature == 0.7

    def test_temperature_one_valid(self):
        model = SingleTableMiniMaxModel(temperature=1.0)
        assert model.temperature == 1.0

    def test_temperature_small_positive_valid(self):
        model = SingleTableMiniMaxModel(temperature=0.01)
        assert model.temperature == 0.01


class TestMiniMaxModelFit:
    """Test model fitting with data and metadata."""

    def test_fit_with_raw_data(
        self, minimax_model: SingleTableMiniMaxModel, raw_data: pd.DataFrame
    ):
        minimax_model.fit(raw_data)
        assert minimax_model.use_raw_data is True
        assert minimax_model.use_metadata is False
        assert len(minimax_model.columns) == len(raw_data.columns)

    def test_fit_with_metadata(
        self,
        minimax_model: SingleTableMiniMaxModel,
        demo_single_table_metadata: Metadata,
    ):
        minimax_model.fit(demo_single_table_metadata)
        assert minimax_model.use_metadata is True
        assert len(minimax_model.columns) > 0

    def test_fit_with_dataloader(
        self,
        minimax_model: SingleTableMiniMaxModel,
        demo_single_table_data_loader: DataLoader,
    ):
        minimax_model.fit(demo_single_table_data_loader)
        assert len(minimax_model.columns) > 0


class TestMiniMaxModelExtraction:
    """Test response extraction (inherits from GPT model)."""

    minimax_response = """
relationship is Husband, fnlwgt is 145441, educational-num is 9, education is HS-grad, occupation is Exec-managerial, gender is Male, race is White, workclass is Private, capital-gain is 0, native-country is United-States, marital-status is Married-civ-spouse, income is >50K, age is 40, capital-loss is 1485, hours-per-week is 40
income is <=50K, gender is Male, education is Assoc-acdm, native-country is ?, educational-num is 12, hours-per-week is 7, occupation is Prof-specialty, capital-gain is 0, capital-loss is 0, fnlwgt is 154164, race is White, workclass is Private, age is 66, relationship is Not-in-family, marital-status is Never-married
"""

    def test_extract_samples(
        self, minimax_model: SingleTableMiniMaxModel, raw_data: pd.DataFrame
    ):
        minimax_model.fit(raw_data)
        features = minimax_model.extract_samples_from_response(self.minimax_response)
        assert isinstance(features, list)
        assert len(features) == 2
        assert len(features[0]) == len(minimax_model.columns)


class TestMiniMaxModelInheritance:
    """Test that MiniMaxModel properly inherits from GPTModel."""

    def test_inherits_from_gpt_model(self):
        from sdgx.models.LLM.single_table.gpt import SingleTableGPTModel

        assert issubclass(SingleTableMiniMaxModel, SingleTableGPTModel)

    def test_client_creation(self, minimax_model: SingleTableMiniMaxModel):
        minimax_model.openai_API_key = "test-key"
        client = minimax_model.openai_client()
        assert client.api_key == "test-key"
        assert "minimax" in str(client.base_url)


# --- Integration Tests (require MINIMAX_API_KEY) ---


MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")


@pytest.mark.skipif(not MINIMAX_API_KEY, reason="MINIMAX_API_KEY not set")
class TestMiniMaxIntegration:
    """Integration tests that call the real MiniMax API."""

    def test_ask_minimax_basic(self, demo_single_table_metadata: Metadata):
        model = SingleTableMiniMaxModel()
        model.fit(demo_single_table_metadata)
        response = model.ask_gpt('Say "test passed" in exactly two words.', model="MiniMax-M2.7")
        assert response is not None
        assert len(response) > 0

    def test_ask_minimax_highspeed(self, demo_single_table_metadata: Metadata):
        model = SingleTableMiniMaxModel()
        model.fit(demo_single_table_metadata)
        response = model.ask_gpt(
            'Say "hello world" in exactly two words.', model="MiniMax-M2.7-highspeed"
        )
        assert response is not None
        assert len(response) > 0

    def test_sample_with_metadata(self, demo_single_table_metadata: Metadata):
        model = SingleTableMiniMaxModel()
        model.query_batch = 5
        model.fit(demo_single_table_metadata)
        result = model.sample(5)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
