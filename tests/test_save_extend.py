import pytest
from pathlib import Path
import pandas as pd
from sdgx.data_models.metadata import Metadata


def test_clean_data():
    df = pd.read_csv("extendBug.csv")
    m = Metadata.from_dataframe(df)
    m.add("Hello", "World")
    n = m.save_extend(Path('clean_data.json'))
    assert n.get("Hello") == ["World"]


def test_int_values():
    df = pd.read_csv("extendBug.csv")
    m = Metadata.from_dataframe(df)
    m.add("Numbers", 55)
    n = m.save_extend(Path('int_values.json'))
    assert n.get("Numbers") == [55]


def test_dict():
    df = pd.read_csv("extendBug.csv")
    m = Metadata.from_dataframe(df)
    m.add("Dict", {"values"})
    n = m.save_extend(Path('dict.json'))
    assert n.get("Dict") == ["values"]


def test_empty():
    df = pd.read_csv("extendBug.csv")
    m = Metadata.from_dataframe(df)
    m.add("", "")
    n = m.save_extend(Path('empty.json'))
    assert n.get("") == [""]


def test_none():
    df = pd.read_csv("extendBug.csv")
    m = Metadata.from_dataframe(df)
    m.add("none", None)
    n = m.save_extend(Path('none.json'))
    assert n.get('none') == [None]


def test_bad_path():
    df = pd.read_csv("extendBug.csv")
    m = Metadata.from_dataframe(df)
    m.add("bad", "path")
    with pytest.raises(AttributeError):
        n = m.save_extend(125)


def test_empty_csv():
    df = pd.read_csv("extendBugEmptyTest.csv")
    with pytest.raises(ZeroDivisionError):
        m = Metadata.from_dataframe(df)
        m.add("empty", "dataframe")
        n = m.save_extend("emptyCSV.json")


def test_bad_dataframe():
    df = None
    with pytest.raises(AttributeError):
        m = Metadata.from_dataframe(df)
        m.add("bad", "dataframe")
        n = m.save_extend("badDataframe.json")