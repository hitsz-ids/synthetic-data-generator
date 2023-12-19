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

    METADATA_SAVE_NAME = "metadata.json"
    MODEL_SAVE_NAME = "model.pkl"

    def __init__(
        self,
        model: str | SynthesizerModel | type[SynthesizerModel],
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
        self.data_processors_manager = DataProcessorManager()
        self.data_processors = [
            d
            if isinstance(d, DataProcessor)
            else self.data_processors_manager.init_data_processor(
                d, **(data_processors_kwargs or {})
            )
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
            self.metadata = Metadata.load(metadata_path)
        else:
            self.metadata = None

        # Init model
        self.model_manager = ModelManager()
        if isinstance(model, SynthesizerModel) and model_path:
            raise SynthesizerInitError(
                "model as instance and model_path cannot be specified at the same time"
            )
        if isinstance(model, str):
            self.model = self.model_manager.init_model(model, **(model_kwargs or {}))
        elif isinstance(model, SynthesizerModel):
            self.model = model
        elif model_path:
            self.model_manager.load(model, model_path)
        else:
            raise SynthesizerInitError("model or model_path must be specified")

        # Other arguments
        self.processored_data_loaders_kwargs = processored_data_loaders_kwargs or {}

    def save(self, save_dir: str | Path) -> Path:
        """
        Dump metadata and model to file
        """
        save_dir = Path(save_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.metadata:
            self.metadata.save(save_dir / self.METADATA_SAVE_NAME)
        self.model.save(save_dir / self.MODEL_SAVE_NAME)
        return save_dir

    @classmethod
    def load(
        cls,
        load_dir: str | Path,
        model: str | type[SynthesizerModel],
        metadata: None | Metadata = None,
        data_connector: None | str | DataConnector = None,
        data_connectors_kwargs: None | dict[str, Any] = None,
        raw_data_loaders_kwargs: None | dict[str, Any] = None,
        processored_data_loaders_kwargs: None | dict[str, Any] = None,
        data_processors: None | list[str | DataProcessor] = None,
        data_processors_kwargs: None | dict[str, dict[str, Any]] = None,
    ) -> "Synthesizer":
        """
        Load metadata and model from file
        """

        load_dir = Path(load_dir).expanduser().resolve()

        if not load_dir.exists():
            raise SynthesizerInitError(f"{load_dir.as_posix()} does not exist")
        model_path = load_dir / cls.MODEL_SAVE_NAME
        if not model_path.exists():
            raise SynthesizerInitError(
                f"{model_path.as_posix()} does not exist, cannot load model."
            )

        metadata_path = load_dir / cls.METADATA_SAVE_NAME
        if not metadata_path.exists():
            metadata_path = None

        return Synthesizer(
            model=model,
            model_path=load_dir / cls.MODEL_SAVE_NAME,
            metadata=metadata,
            metadata_path=metadata_path,
            data_connector=data_connector,
            data_connectors_kwargs=data_connectors_kwargs,
            raw_data_loaders_kwargs=raw_data_loaders_kwargs,
            processored_data_loaders_kwargs=processored_data_loaders_kwargs,
            data_processors=data_processors,
            data_processors_kwargs=data_processors_kwargs,
        )

    def fit(
        self,
        metadata: None | Metadata = None,
        inspector_max_chunk: int = 10,
        metadata_include_inspectors: None | list[str] = None,
        metadata_exclude_inspectors: None | list[str] = None,
        inspector_init_kwargs: None | dict[str, Any] = None,
        model_fit_kwargs: None | dict[str, Any] = None,
    ):
        if self.dataloader is None:
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
                    chunk = d.convert(chunk)
                yield chunk

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

        def generator_sample_caller():
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

        return generator_sample_caller()

    def cleanup(self):
        if self.dataloader:
            self.dataloader.finalize(clear_cache=True)
