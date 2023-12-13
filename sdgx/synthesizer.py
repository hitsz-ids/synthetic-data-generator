from __future__ import annotations

from pathlib import Path
from typing import Any

from sdgx.data_connectors.base import DataConnector
from sdgx.data_connectors.manager import DataConnectorManager
from sdgx.data_loader import DataLoader
from sdgx.data_processors.base import DataProcessor
from sdgx.data_processors.manager import DataProcessorManager
from sdgx.exceptions import SynthesizerInitError
from sdgx.models.base import SynthesizerModel
from sdgx.models.manager import ModelManager


class Synthesizer:
    """
    Synthesizer takes all logic of preprocessing data, fit model and sample
    """

    def __init__(
        self,
        data_connector: str | DataConnector,
        data_processors: list[str | DataProcessor],
        model: str | SynthesizerModel,
        model_path: None | str | Path = None,
        data_connectors_kwargs: None | dict[str, Any] = None,
        data_loaders_kwargs: None | dict[str, Any] = None,
        data_processors_kwargs: None | dict[str, dict[str, Any]] = None,
        model_kwargs: None | dict[str, Any] = None,
    ):
        if isinstance(data_connector, str):
            data_connector = DataConnectorManager().init_data_connector(
                data_connector, **(data_connectors_kwargs or {})
            )
        self.dataloader = DataLoader(
            data_connector,
            **(data_loaders_kwargs or {}),
        )

        self.data_processors = [
            d
            if isinstance(d, DataProcessor)
            else DataProcessorManager().init_data_processor(d, **(data_processors_kwargs or {}))
            for d in data_processors
        ]

        if model and model_path:
            raise SynthesizerInitError("model and model_path cannot be specified at the same time")

        if isinstance(model, str):
            self.model = ModelManager().init_model(model, **(model_kwargs or {}))
        elif isinstance(model, SynthesizerModel):
            self.model = model
        elif model_path:
            model_path = Path(model_path).expanduser().resolve()
            self.model = ModelManager().load(model_path)
        else:
            raise SynthesizerInitError("model or model_path must be specified")
