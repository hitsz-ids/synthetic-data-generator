from __future__ import annotations

import threading
import urllib.request
from pathlib import Path

import pandas as pd

from sdgx.log import logger

try:
    from functools import cache
except ImportError:
    from functools import lru_cache as cache

__all__ = ["download_demo_data", "get_demo_single_table", "cache", "Singleton"]


def download_demo_data(data_dir: str | Path = "./dataset") -> Path:
    """
    Download demo data if not exist

    Args:
        data_dir(str | Path): data directory

    Returns:
        pathlib.Path: demo data path
    """
    data_dir = Path(data_dir).expanduser().resolve()
    demo_data_path = data_dir / "adult.csv"
    if not demo_data_path.exists():
        # Download from datahub
        demo_data_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading demo data from datahub.io to {}".format(demo_data_path))
        url = "https://datahub.io/machine-learning/adult/r/adult.csv"
        urllib.request.urlretrieve(url, demo_data_path)

    return demo_data_path


def get_demo_single_table(data_dir: str | Path = "./dataset"):
    """
    Get demo single table as DataFrame and discrete columns names

    Args:
        data_dir(str | Path): data directory

    Returns:

        pd.DataFrame: demo single table
        list: discrete columns
    """
    demo_data_path = download_demo_data(data_dir)
    pd_obj = pd.read_csv(demo_data_path)
    discrete_cols = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "class",
    ]
    return pd_obj, discrete_cols


class Singleton(type):
    """
    metaclass for singleton, thread-safe.
    """

    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
