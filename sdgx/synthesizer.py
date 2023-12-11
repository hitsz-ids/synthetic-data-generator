from __future__ import annotations

from typing import Any

from sdgx.data_connectors.base import DataConnector
from sdgx.data_processors.base import DataProcessor
from sdgx.models.base import SynthesizerModel


class Synthesizer:
    """
    Synthesizer takes all logic of preprocessing data, fit model and sample
    """

    def __init__(
        self,
        data_connectors: str | DataConnector,
        data_processors: list[str | DataProcessor],
        model: SynthesizerModel,
        data_connectors_kwargs: None | dict[str, Any] = None,
        data_processors_args: None | dict[str, dict[str, Any]] = None,
    ):
        pass
