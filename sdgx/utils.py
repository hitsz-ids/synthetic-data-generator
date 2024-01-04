from __future__ import annotations

import socket
import threading
import urllib.request
from contextlib import closing
from pathlib import Path

import pandas as pd

from sdgx.log import logger
from sdgx.data_models.relationship import Relationship
# from sdgx.data_models.metadata import Metadata
# from sdgx.data_models.combiner import MetadataCombiner

try:
    from functools import cache
except ImportError:
    from functools import lru_cache as cache

__all__ = ["download_demo_data", "get_demo_single_table", "cache", "Singleton", "find_free_port",
           "download_multi_table_demo_data", "get_demo_single_table"]

MULTI_TABLE_DEMO_DATA = {
    "rossman":{
        "parent_table": "store", 
        "child_table" : "train", 
        "parent_url": "https://raw.githubusercontent.com/juniorcl/rossman-store-sales/main/databases/store.csv",
        "child_url": "https://raw.githubusercontent.com/juniorcl/rossman-store-sales/main/databases/train.csv",
        "parent_primary_keys": ["Store"],
        "child_primary_keys" : ["Store", "Date"],
        "foreign_keys" : ["Store"]
    }
}



def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


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


def download_multi_table_demo_data(data_dir: str | Path = "./dataset", dataset_name = "rossman") -> dict[str, Path]:
    """
    Download multi-table demo data "Rossman Store Sales" or "Rossmann Store Sales" if not exist

    Args:
        data_dir(str | Path): data directory

    Returns:
        dict[str, pathlib.Path]: dict, the key is table name, value is demo data path
    """
    demo_data_info = MULTI_TABLE_DEMO_DATA[dataset_name]
    data_dir = Path(data_dir).expanduser().resolve()
    parent_file_name = dataset_name + "_" + demo_data_info["parent_table"] + '.csv'
    child_file_name  = dataset_name + "_" + demo_data_info["child_table"] + '.csv'
    demo_data_path_parent = data_dir / parent_file_name
    demo_data_path_child  = data_dir / child_file_name
    # For now, I think it's OK to hardcode the URL for each dataset
    # In the future we can consider using our own S3 Bucket or providing more data sets through sdg.idslab.io. 
    if not demo_data_path_parent.exists():
        # make dir
        demo_data_path_parent.parent.mkdir(parents=True, exist_ok=True)
        # download parent table from github link
        logger.info("Downloading parent table from github to {}".format(demo_data_path_parent))
        parent_url = demo_data_info["parent_url"]
        urllib.request.urlretrieve(parent_url, demo_data_path_parent)
    # then child table
    if not demo_data_path_child.exists():
        # make dir
        demo_data_path_child.parent.mkdir(parents=True, exist_ok=True)
        # download child table from github link
        logger.info("Downloading child table from github to {}".format(demo_data_path_child))
        parent_url = demo_data_info["child_url"]
        urllib.request.urlretrieve(parent_url, demo_data_path_child)
    
    return {demo_data_info["parent_table"]: demo_data_path_parent,
            demo_data_info["child_table" ]: demo_data_path_child  }


def get_demo_multi_table(data_dir: str | Path = "./dataset", dataset_name = "rossman"):
    """
    Get multi-table demo data as DataFrame and relationship

    Args:
        data_dir(str | Path): data directory

    Returns:

        dict[str, pd.DataFrame]: multi-table data dict, the key is table name, value is DataFrame.
        MetadataCombiner: metadata and relationships about demo data tables.
    """
    demo_data_info = MULTI_TABLE_DEMO_DATA[dataset_name]
    multi_table_dict = {}
    # download if not exist 
    demo_data_dict = download_multi_table_demo_data(data_dir, dataset_name)
    # read 
    for table_name in demo_data_dict.keys():
        each_path = demo_data_dict[table_name]
        pd_obj = pd.read_csv(each_path)
        multi_table_dict[table_name] = pd_obj
    # build relationship 
    demo_relationship = Relationship.build(
        parent_table = demo_data_info["parent_table"],
        child_table  = demo_data_dict["child_table" ],
        foreign_keys = demo_data_dict["foreign_keys"]
    )
    # get metadata 
    # parent_metadata = Metadata.from_dataframe(demo_data_dict[demo_data_info["parent_table"]])
    # child_metadata  = Metadata.from_dataframe(demo_data_dict[demo_data_info["child_table" ]])

    
    return multi_table_dict
