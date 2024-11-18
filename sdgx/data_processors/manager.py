from __future__ import annotations

from typing import Any

from sdgx import data_processors
from sdgx.data_processors import extension
from sdgx.data_processors.base import DataProcessor
from sdgx.data_processors.extension import project_name as PROJECT_NAME
from sdgx.manager import Manager


class DataProcessorManager(Manager):
    """
    This is a plugin management class for data processing components.

    Properties:
        - register_type: Specifies the type of data processors to register.
        - project_name: Stores the project name from the extension module.
        - hookspecs_model: Stores the hook specifications model from the extension module.
        - preset_default_processors: Stores a list of default processor names in lowercase.
        - registed_data_processors: Property that returns the registered data processors.
        - registed_default_processor_list: Property that returns the registered default data processors.

    Methods:
        - load_all_local_model: Loads all local models for formatters, generators, samplers, and transformers.
        - init_data_processor: Initializes a data processor with the given name and keyword arguments.
        - init_all_processors: Initializes all registered data processors with optional keyword arguments.
        - init_default_processors: Initializes default processors that are both registered and preset.

    """

    register_type = DataProcessor
    """
    Specifies the type of data processors to register."""

    project_name = PROJECT_NAME
    """
    Stores the project name from the extension module.
    """

    hookspecs_model = extension
    """
    The hook specifications model from the extension module.
    """

    preset_defalut_processors = [
        p.lower()
        for p in [
            "SpecificCombinationTransformer",
            "FixedCombinationTransformer",
            "NonValueTransformer",
            "OutlierTransformer",
            "EmailGenerator",
            "ChnPiiGenerator",
            "IntValueFormatter",
            "DatetimeFormatter",
        ]
    ] + [
        "ConstValueTransformer".lower(),
        "PositiveNegativeFilter".lower(),
        "EmptyTransformer".lower(),
        "ColumnOrderTransformer".lower(),
    ]
    """
    preset_defalut_processors list stores the lowercase names of the transformers loaded by default. When using the synthesizer, they will be loaded by default to facilitate user operations.

    Keep ColumnOrderTransformer always at the last one.
    """

    @property
    def registed_data_processors(self):
        """
        This property returns all registered data processors
        """
        return self.registed_cls

    @property
    def registed_default_processor_list(self):
        """
        This property returns all registered default data processors
        """
        registed_processor_list = self.registed_data_processors.keys()

        # we donot use the code next line to calculate processor intersection, to ensure the oder of processors not changed.
        # target_processors = list(set(registed_processor_list).intersection(self.preset_defalut_processors))

        # calculate the target_processors
        default_processors = []
        for each_processor in self.preset_defalut_processors:
            if each_processor in registed_processor_list:
                default_processors.append(each_processor)

        return default_processors

    def load_all_local_model(self):
        """
        loads all local models
        """
        self._load_dir(data_processors.formatters)
        self._load_dir(data_processors.generators)
        self._load_dir(data_processors.samplers)
        self._load_dir(data_processors.transformers)
        self._load_dir(data_processors.filter)

    def init_data_processor(self, processor_name, **kwargs: dict[str, Any]) -> DataProcessor:
        """
        Initializes a data processor with the given name and parameters
        """
        return self.init(processor_name, **kwargs)

    def init_all_processors(self, **kwargs: Any) -> list[DataProcessor]:
        """
        Initializes all registered data processors
        """
        return [
            self.init(processor_name, **kwargs)
            for processor_name in self.registed_data_processors.keys()
        ]

    def init_default_processors(self, **kwargs: Any) -> list[DataProcessor]:
        """
        Initializes all default data processors
        """

        return [
            self.init(processor_name, **kwargs)
            for processor_name in self.registed_default_processor_list
        ]
