from __future__ import annotations

import time
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
from sdgx.exceptions import SynthesizerInitError, SynthesizerSampleError
from sdgx.models.base import SynthesizerModel
from sdgx.models.manager import ModelManager
from sdgx.models.statistics.single_table.base import StatisticSynthesizerModel
from sdgx.utils import logger


class Synthesizer:
    """
    Synthesizer is the high level interface for synthesizing data.

    We provided several example usage in our `Github repository <https://github.com/hitsz-ids/synthetic-data-generator/tree/main/example>`_.

    Args:

        model (str | SynthesizerModel | type[SynthesizerModel]): The name of the model or the model itself. Type of model must be :class:`~sdgx.models.base.SynthesizerModel`.
            When model is a string, it must be registered in :class:`~sdgx.models.manager.ModelManager`.
        model_path (str | Path, optional): The path to the model file. Defaults to None. Used to load the model if ``model`` is a string or type of :class:`~sdgx.models.base.SynthesizerModel`.
        model_kwargs (dict[str, Any], optional): The keyword arguments for model. Defaults to None.
        metadata (Metadata, optional): The metadata to use. Defaults to None.
        metadata_path (str | Path, optional): The path to the metadata file. Defaults to None. Used to load the metadata if ``metadata`` is None.
        data_connector (DataConnector | type[DataConnector] | str, optional): The data connector to use. Defaults to None.
            When data_connector is a string, it must be registered in :class:`~sdgx.data_connectors.manager.DataConnectorManager`.
        data_connector_kwargs (dict[str, Any], optional): The keyword arguments for data connectors. Defaults to None.
        raw_data_loaders_kwargs (dict[str, Any], optional): The keyword arguments for raw data loaders. Defaults to None.
        processed_data_loaders_kwargs (dict[str, Any], optional): The keyword arguments for processed data loaders. Defaults to None.
        data_processors (list[str | DataProcessor | type[DataProcessor]], optional): The data processors to use. Defaults to None.
            When data_processor is a string, it must be registered in :class:`~sdgx.data_processors.manager.DataProcessorManager`.
        data_processors_kwargs (dict[str, dict[str, Any]], optional): The keyword arguments for data processors. Defaults to None.

    Example:

        .. code-block:: python

            from sdgx.data_connectors.csv_connector import CsvConnector
            from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
            from sdgx.synthesizer import Synthesizer
            from sdgx.utils import download_demo_data

            dataset_csv = download_demo_data()
            data_connector = CsvConnector(path=dataset_csv)
            synthesizer = Synthesizer(
                model=CTGANSynthesizerModel(epochs=1),  # For quick demo
                data_connector=data_connector,
            )
            synthesizer.fit()
            sampled_data = synthesizer.sample(1000)
    """

    METADATA_SAVE_NAME = "metadata.json"
    """
    Default name for metadata file
    """
    MODEL_SAVE_DIR = "model"
    """
    Default name for model directory
    """

    def __init__(
        self,
        model: str | SynthesizerModel | type[SynthesizerModel],
        model_path: None | str | Path = None,
        model_kwargs: None | dict[str, Any] = None,
        metadata: None | Metadata = None,
        metadata_path: None | str | Path = None,
        data_connector: None | str | DataConnector | type[DataConnector] = None,
        data_connector_kwargs: None | dict[str, Any] = None,
        raw_data_loaders_kwargs: None | dict[str, Any] = None,
        processed_data_loaders_kwargs: None | dict[str, Any] = None,
        data_processors: None | list[str | DataProcessor | type[DataProcessor]] = None,
        data_processors_kwargs: None | dict[str, Any] = None,
    ):
        # Init data connectors
        if isinstance(data_connector, str) or isinstance(data_connector, type):
            data_connector = DataConnectorManager().init_data_connector(
                data_connector, **(data_connector_kwargs or {})
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
        self.data_processors_manager = DataProcessorManager()
        if not data_processors:
            data_processors = self.data_processors_manager.registed_default_processor_list
        logger.info(f"Using data processors: {data_processors}")
        self.data_processors = [
            (
                d
                if isinstance(d, DataProcessor)
                else self.data_processors_manager.init_data_processor(
                    d, **(data_processors_kwargs or {})
                )
            )
            for d in data_processors
        ]
        if metadata and metadata_path:
            raise SynthesizerInitError(
                "metadata and metadata_path cannot be specified at the same time"
            )

        # Load metadata
        # metadata also can be changed in ``fit`` or ``sample``
        # Always use the latest metadata configured.
        if metadata:
            self.metadata = metadata
        elif metadata_path:
            self.metadata = Metadata.load(metadata_path)
        else:
            self.metadata = None

        # Init model
        self.model_manager = ModelManager()
        if isinstance(model, SynthesizerModel) and model_path:
            # Initialized model cannot load from model_path
            raise SynthesizerInitError(
                "model as instance and model_path cannot be specified at the same time"
            )
        if (isinstance(model, str) or isinstance(model, type)) and model_path:
            # Load model by cls or str
            self.model = self.model_manager.load(model, model_path)
            if model_kwargs:
                logger.warning("model_kwargs will be ignored when loading model from model_path")
        elif isinstance(model, str) or isinstance(model, type):
            # Init model by cls or str
            self.model = self.model_manager.init_model(model, **(model_kwargs or {}))
        elif isinstance(model, SynthesizerModel) or isinstance(model, StatisticSynthesizerModel):
            # Already initialized model
            self.model = model
            if model_kwargs:
                logger.warning("model_kwargs will be ignored when using already initialized model")
        else:
            raise SynthesizerInitError("model or model_path must be specified")

        # Other arguments
        self.processed_data_loaders_kwargs = processed_data_loaders_kwargs or {}

    def save(self, save_dir: str | Path) -> Path:
        """
        Dump metadata and model to file

        Args:
            save_dir (str | Path): The directory to save the model.

        Returns:
            Path: The directory to save the synthesizer.
        """
        save_dir = Path(save_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving synthesizer to {save_dir}")

        if self.metadata:
            self.metadata.save(save_dir / self.METADATA_SAVE_NAME)

        model_save_dir = save_dir / self.MODEL_SAVE_DIR
        model_save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(model_save_dir)
        return save_dir

    @classmethod
    def load(
        cls,
        load_dir: str | Path,
        model: str | type[SynthesizerModel],
        metadata: None | Metadata = None,
        data_connector: None | str | DataConnector | type[DataConnector] = None,
        data_connector_kwargs: None | dict[str, Any] = None,
        raw_data_loaders_kwargs: None | dict[str, Any] = None,
        processed_data_loaders_kwargs: None | dict[str, Any] = None,
        data_processors: None | list[str | DataProcessor | type[DataProcessor]] = None,
        data_processors_kwargs: None | dict[str, dict[str, Any]] = None,
    ) -> "Synthesizer":
        """
        Load metadata and model, allow rebuilding Synthesizer for finetuning or other use cases.

        We need ``model`` as not every model support *pickle* way to save and load.

        Args:
            load_dir (str | Path): The directory to load the model.
            model (str | type[SynthesizerModel]): The name of the model or the model itself. Type of model must be :class:`~sdgx.models.base.SynthesizerModel`.
                When model is a string, it must be registered in :class:`~sdgx.models.manager.ModelManager`.
            metadata (Metadata, optional): The metadata to use. Defaults to None.
            data_connector (DataConnector | type[DataConnector] | str, optional): The data connector to use. Defaults to None.
                When data_connector is a string, it must be registered in :class:`~sdgx.data_connectors.manager.DataConnectorManager`.
            data_connector_kwargs (dict[str, Any], optional): The keyword arguments for data connectors. Defaults to None.
            raw_data_loaders_kwargs (dict[str, Any], optional): The keyword arguments for raw data loaders. Defaults to None.
            processed_data_loaders_kwargs (dict[str, Any], optional): The keyword arguments for processed data loaders. Defaults to None.
            data_processors (list[str | DataProcessor | type[DataProcessor]], optional): The data processors to use. Defaults to None.
                When data_processor is a string, it must be registered in :class:`~sdgx.data_processors.manager.DataProcessorManager`.
            data_processors_kwargs (dict[str, dict[str, Any]], optional): The keyword arguments for data processors. Defaults to None.

        Returns:
            Synthesizer: The synthesizer instance.
        """

        load_dir = Path(load_dir).expanduser().resolve()
        logger.info(f"Loading synthesizer from {load_dir}")

        if not load_dir.exists():
            raise SynthesizerInitError(f"{load_dir.as_posix()} does not exist")
        model_path = load_dir / cls.MODEL_SAVE_DIR
        if not model_path.exists():
            raise SynthesizerInitError(
                f"{model_path.as_posix()} does not exist, cannot load model."
            )

        metadata_path = load_dir / cls.METADATA_SAVE_NAME
        if not metadata_path.exists():
            metadata_path = None

        return Synthesizer(
            model=model,
            model_path=model_path,
            metadata=metadata,
            metadata_path=metadata_path,
            data_connector=data_connector,
            data_connector_kwargs=data_connector_kwargs,
            raw_data_loaders_kwargs=raw_data_loaders_kwargs,
            processed_data_loaders_kwargs=processed_data_loaders_kwargs,
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
        """
        Fit the synthesizer with metadata and data processors.

        Raw data will be loaded from the dataloader and processed by the data processors in a Generator.
        The Generator, which prevents the processed data, will be wrapped into a DataLoader, aka ProcessedDataLoader.
        The ProcessedDataLoader will be used to fit the model.

        For more information about DataLoaders, please refer to the :class:`~sdgx.data_loaders.base.DataLoader`.

        For more information about DataProcessors, please refer to the :class:`~sdgx.data_processors.base.DataProcessor`.

        For more information about DataConnectors, please refer to the :class:`~sdgx.data_connectors.base.DataConnector`. Especially, the :class:`~sdgx.data_connectors.generator_connector.GeneratorConnector`.

        Args:
            metadata (Metadata, optional): The metadata to use. Defaults to None. If None, it will be inferred from the dataloader with the :func:`~sdgx.data_models.metadata.Metadata.from_dataloader` method.
            inspector_max_chunk (int, optional): The maximum number of chunks to inspect. Defaults to 10.
            metadata_include_inspectors (list[str], optional): The list of metadata inspectors to include. Defaults to None.
            metadata_exclude_inspectors (list[str], optional): The list of metadata inspectors to exclude. Defaults to None.
            inspector_init_kwargs (dict[str, Any], optional): The keyword arguments for metadata inspectors. Defaults to None.
            model_fit_kwargs (dict[str, Any], optional): The keyword arguments for model.fit. Defaults to None.
        """
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
        self.metadata = metadata  # Ensure update metadata

        logger.info("Fitting data processors...")
        if not self.dataloader:
            logger.info("Fitting without dataloader.")
        start_time = time.time()
        for d in self.data_processors:
            if self.dataloader:
                d.fit(metadata=metadata, tabular_data=self.dataloader)
            else:
                d.fit(metadata=metadata)
        logger.info(
            f"Fitted {len(self.data_processors)} data processors in  {time.time() - start_time}s."
        )

        def chunk_generator() -> Generator[pd.DataFrame, None, None]:
            for chunk in self.dataloader.iter():
                for d in self.data_processors:
                    chunk = d.convert(chunk)
                yield chunk

        logger.info("Initializing processed data loader...")
        start_time = time.time()
        processed_dataloader = DataLoader(
            GeneratorConnector(chunk_generator),
            identity=self.dataloader.identity,
            **self.processed_data_loaders_kwargs,
        )
        logger.info(f"Initialized processed data loader in {time.time() - start_time}s")
        try:
            logger.info("Model fit Started...")
            self.model.fit(metadata, processed_dataloader, **(model_fit_kwargs or {}))
            logger.info("Model fit... Finished")
        finally:
            processed_dataloader.finalize(clear_cache=True)

    def sample(
        self,
        count: int,
        chunksize: None | int = None,
        metadata: None | Metadata = None,
        model_sample_args: None | dict[str, Any] = None,
    ) -> pd.DataFrame | Generator[pd.DataFrame, None, None]:
        """
        Sample data from the synthesizer.

        Args:
            count (int): The number of samples to generate.
            chunksize (int, optional): The chunksize to use. Defaults to None. If is not None, the data will be sampled in chunks.
                And will return a generator that yields chunks of samples.
            metadata (Metadata, optional): The metadata to use. Defaults to None. If None, will use the metadata in fit first.
            model_sample_args (dict[str, Any], optional): The keyword arguments for model.sample. Defaults to None.

        Returns:
            pd.DataFrame | typing.Generator[pd.DataFrame, None, None]: The sampled data. When chunksize is not None, it will be a generator.
        """
        logger.info("Sampling...")
        metadata = metadata or self.metadata
        self.metadata = metadata  # Ensure update metadata

        # data_processors do not need to be fit again in the sampling stage

        if not model_sample_args:
            model_sample_args = {}

        if chunksize is None:
            return self._sample_once(count, model_sample_args)

        if chunksize > count:
            raise SynthesizerSampleError("chunksize must be less than or equal to count")

        def generator_sample_caller():
            sample_times = count // chunksize
            for _ in range(sample_times):
                sample_data = self._sample_once(chunksize, model_sample_args)
                for d in self.data_processors:
                    sample_data = d.reverse_convert(sample_data)
                yield sample_data

            if count % chunksize > 0:
                sample_data = self._sample_once(count % chunksize, model_sample_args)
                for d in self.data_processors:
                    sample_data = d.reverse_convert(sample_data)
                yield sample_data

        return generator_sample_caller()

    def _sample_once(
        self, count: int, model_sample_args: None | dict[str, Any] = None
    ) -> pd.DataFrame:
        """
        Sample data once.

        DataProcessors may drop some broken data after reverse_convert.
        So we oversample first and then take the first `count` samples.

        TODO:

            - Use an adaptive scale for oversampling will be better for performance.

        """
        missing_count = count
        max_trails = 50
        sample_data_list = []
        while missing_count > 0 and max_trails > 0:
            sample_data = self.model.sample(int(missing_count * 4), **model_sample_args)
            for d in self.data_processors:
                sample_data = d.reverse_convert(sample_data)
            sample_data = sample_data.dropna(how="all")
            sample_data_list.append(sample_data)
            missing_count = missing_count - len(sample_data)
            max_trails -= 1

        return pd.concat(sample_data_list)[:count]

    def cleanup(self):
        """
        Cleanup resources. This will cause model unavailable and clear the cache.

        It useful when Synthesizer object is no longer needed and may hold large resources like GPUs.
        """

        if self.dataloader:
            self.dataloader.finalize(clear_cache=True)
        # Release resources
        if hasattr(self, "model"):
            del self.model

    def __del__(self):
        self.cleanup()
