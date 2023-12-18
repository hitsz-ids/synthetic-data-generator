"""Functions to profile performance of RDT Transformers."""

# pylint: disable=W0212

import multiprocessing as mp
import timeit
import tracemalloc
from copy import deepcopy

import pandas as pd


def _profile_time(transformer, method_name, dataset, column=None, iterations=10, copy=False):
    total_time = 0
    for _ in range(iterations):
        if copy:
            transformer_copy = deepcopy(transformer)
            method = getattr(transformer_copy, method_name)

        else:
            method = getattr(transformer, method_name)

        start_time = timeit.default_timer()
        if column:
            method(dataset, column)
        else:
            method(dataset)
        total_time += timeit.default_timer() - start_time

    return total_time / iterations


def _set_memory_for_method(method, dataset, column, peak_memory):
    tracemalloc.start()
    if column:
        method(dataset, column)
    else:
        method(dataset)

    peak_memory.value = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    tracemalloc.clear_traces()


def _profile_memory(method, dataset, column=None):
    ctx = mp.get_context("spawn")
    peak_memory = ctx.Value("i", 0)
    profiling_process = ctx.Process(
        target=_set_memory_for_method, args=(method, dataset, column, peak_memory)
    )
    profiling_process.start()
    profiling_process.join()
    return peak_memory.value


def profile_transformer(transformer, dataset_generator, transform_size, fit_size=None):
    """Profile a Transformer on a dataset.

    This function will get the total time and peak memory
    for the ``fit``, ``transform`` and ``reverse_transform``
    methods of the provided transformer against the provided
    dataset.

    Args:
        transformer (Transformer):
            Transformer instance.
        dataset_generator (DatasetGenerator):
            DatasetGenerator instance.
        transform_size (int):
            Number of rows to generate for ``transform`` and ``reverse_transform``.
        fit_size (int or None):
            Number of rows to generate for ``fit``. If None, use ``transform_size``.

    Returns:
        pandas.Series:
            Series containing the time and memory taken by ``fit``, ``transform``,
            and ``reverse_transform`` for the transformer.
    """
    fit_size = fit_size or transform_size
    fit_dataset = pd.DataFrame({"test": dataset_generator.generate(fit_size)})
    replace = transform_size > fit_size
    transform_dataset = fit_dataset.sample(transform_size, replace=replace)

    fit_time = _profile_time(transformer, "fit", fit_dataset, column="test", copy=True)
    fit_memory = _profile_memory(transformer.fit, fit_dataset, column="test")
    transformer.fit(fit_dataset, "test")

    transform_time = _profile_time(transformer, "transform", transform_dataset)
    transform_memory = _profile_memory(transformer.transform, transform_dataset)

    reverse_dataset = transformer.transform(transform_dataset)
    reverse_time = _profile_time(transformer, "reverse_transform", reverse_dataset)
    reverse_memory = _profile_memory(transformer.reverse_transform, reverse_dataset)

    return pd.Series(
        {
            "Fit Time": fit_time,
            "Fit Memory": fit_memory,
            "Transform Time": transform_time,
            "Transform Memory": transform_memory,
            "Reverse Transform Time": reverse_time,
            "Reverse Transform Memory": reverse_memory,
        }
    )
