from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

import pandas as pd

from sdgx.data_connectors.base import DataConnector
from sdgx.data_connectors.generator_connector import GeneratorConnector
from sdgx.data_connectors.manager import DataConnectorManager
from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.base import DataProcessor
from sdgx.data_processors.manager import DataProcessorManager
from sdgx.exceptions import SynthesizerInitError
from sdgx.log import logger
from sdgx.models.base import SynthesizerModel
from sdgx.models.manager import ModelManager


class Synthesizer:
    """
    Synthesizer takes all logic of preprocessing data, fit model and sample
    """

    def __init__(
        self,
        model: str | SynthesizerModel,
        model_path: None | str | Path = None,
        metadata: None | Metadata = None,
        metadata_path: None | str | Path = None,
        data_connector: None | str | DataConnector = None,
        data_connectors_kwargs: None | dict[str, Any] = None,
        raw_data_loaders_kwargs: None | dict[str, Any] = None,
        processored_data_loaders_kwargs: None | dict[str, Any] = None,
        data_processors: None | list[str | DataProcessor] = None,
        data_processors_kwargs: None | dict[str, dict[str, Any]] = None,
        model_kwargs: None | dict[str, Any] = None,
    ):
        # Init data connectors
        if isinstance(data_connector, str):
            data_connector = DataConnectorManager().init_data_connector(
                data_connector, **(data_connectors_kwargs or {})
            )
        if data_connector:
            self.dataloader = DataLoader(
                data_connector,
                **(raw_data_loaders_kwargs or {}),
            )
        else:
            logger.warning("No data_connector provided, will not support `fit`")
            self.dataloader = None

        # Init data processors
        if not data_processors:
            data_processors = []
        self.data_processors = [
            d
            if isinstance(d, DataProcessor)
            else DataProcessorManager().init_data_processor(d, **(data_processors_kwargs or {}))
            for d in data_processors
        ]
        if metadata and metadata_path:
            raise SynthesizerInitError(
                "metadata and metadata_path cannot be specified at the same time"
            )

        # Load metadata
        if metadata:
            self.metadata = metadata
        elif metadata_path:
            self.metadata.load(metadata_path)
        else:
            self.metadata = None

        # Init model
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

        # Other arguments
        self.processored_data_loaders_kwargs = processored_data_loaders_kwargs or {}

    def save(self) -> Path:
        """
        TODO: Dump metadata and model to file
        """

    @classmethod
    def load(cls) -> Synthesizer:
        """
        TODO: Load metadata and model from file

        Loaded Synthesizer only support sample
        """

    def fit(
        self,
        metadata: None | Metadata = None,
        inspector_max_chunk: int = 10,
        metadata_include_inspectors: None | list[str] = None,
        metadata_exclude_inspectors: None | list[str] = None,
        inspector_init_kwargs: None | dict[str, Any] = None,
        model_fit_kwargs: None | dict[str, Any] = None,
    ):
        if not self.dataloader:
            raise SynthesizerInitError(
                "Cannot fit without dataloader, check `data_connector` parameter when initializing Synthesizer"
            )

        metadata = (
            metadata
            or self.metadata
            or Metadata.from_dataloader(
                self.dataloader,
                max_chunk=inspector_max_chunk,
                include_inspectors=metadata_include_inspectors,
                exclude_inspectors=metadata_exclude_inspectors,
                inspector_init_kwargs=inspector_init_kwargs,
            )
        )
        for d in self.data_processors:
            d.fit(metadata)

        def chunk_generator() -> Generator[pd.DataFrame, None, None]:
            for chunk in self.dataloader.iter():
                for d in self.data_processors:
                    yield d.convert(chunk)

        processed_dataloader = DataLoader(
            GeneratorConnector(chunk_generator),
            **self.processored_data_loaders_kwargs,
        )
        try:
            self.model.fit(metadata, processed_dataloader, **(model_fit_kwargs or {}))
        finally:
            processed_dataloader.finalize(clear_cache=True)

    def sample(
        self,
        count: int,
        chunksize: None | int = None,
        metadata: None | Metadata = None,
        model_fit_kwargs: None | dict[str, Any] = None,
    ) -> pd.DataFrame | Generator[pd.DataFrame, None, None]:
        metadata = metadata or self.metadata
        if metadata:
            for d in self.data_processors:
                d.fit(metadata)
        if not model_fit_kwargs:
            model_fit_kwargs = {}

        if chunksize is None:
            sample_data = self.model.sample(count, **model_fit_kwargs)
            for d in self.data_processors:
                sample_data = d.reverse_convert(sample_data)
            return sample_data

        sample_times = count // chunksize
        for _ in range(sample_times):
            sample_data = self.model.sample(chunksize, **model_fit_kwargs)
            for d in self.data_processors:
                sample_data = d.reverse_convert(sample_data)
            yield sample_data

        if count % chunksize > 0:
            sample_data = self.model.sample(count % chunksize, **model_fit_kwargs)
            for d in self.data_processors:
                sample_data = d.reverse_convert(sample_data)
            yield sample_data
