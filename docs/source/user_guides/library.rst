Use Synthetic Data Generator as a library
==================================================

.. Note::

    Learn more about :ref:`Architecture <architecture>` of our project.


.. Warming::

    This guide is not complete yet. Welcome to contribute.

Use SDG as a library allow researchers or developers to build their own project
based on SDG. It's highly recommended to use SDG as a library if people have some
basic programming experience.

All avaliable built-in conponents are listed in :ref:`API Reference <api_reference>`.
You can also extend SDG with your own components, see :ref:`Developer guides for extension <Extented Synthetic Data Generator>` for more details.


Use :ref:`Data Connector <DataConnector>` to connect data resources.
---------------------------------------------------------------------------------

``Data Connector`` provide a unified interface to read data from different formats or
data sources. Avaliable data connectors are listed in :ref:`Built-in Data Connectors <Built-in DataConnector>`.

.. code-block:: python

    # Create data connector for csv file
    from sdgx.data_connectors.csv_connector import CsvConnector

    dataset_csv = "path/to/dataset.csv"
    data_connector = CsvConnector(path=dataset_csv)

Use :ref:`Data Processor <DataProcessor>` to preprocess data.
---------------------------------------------------------------------------------

``Data Processor`` is a powerful tool to process and transform data before fit model,
and post-processor data. And it is closely associated with :ref:`Metadata <metadata>`.

People can use ``Data Processor`` for following tasks:

- Formatting one or more columns from one data type to another, e.g. datetime to timestamp
- Generate reliable data though bypass models, e.g. generating reliable address through `Faker <https://github.com/joke2k/faker/>`_.
- Masking sensitive information
- Discarding non-compliant data

We provide built-in data processors in :ref:`Built-in Data Processors <Built-in DataProcessor>`.

.. TODO: Data processor has not been implemented yet.
.. code-block:: python

    # Create datetime formatter
    # from sdgx.data_processors.datetime_formatter import DatetimeFormatter


Customize :ref:`Metadata <metadata>` for your dataset
---------------------------------------------------------------------------------

SDG has a built-in metadata :ref:`Inspector <inspectors>` for metadata inspection.
It is convenient but not always accurate for your dataset.

So you can modify the metadata of your dataset before fit model.


.. TODO: Metadata has not been implemented yet.
.. code-block:: python

    from sdgx.data_models.metadata import Metadata
    metadata = Metadata.from_dataframe(df)


Use :ref:`Synthesizer <Synthesizer>` to generate synthetic data
---------------------------------------------------------------------------------

Synthesizer is the high level interface for synthesizing data.
It combines all components above and use serveral steps to generate synthetic data.

There are lots of models in :ref:`Built-in Models <Built-in Models>`,
and you can also use your own models.

.. Note::

    :ref:`DataLoader <api_reference/data_loader>` and :ref:`Cacher for DataLoader <api_reference/cachers-extension>` are used in synthesizer.
    They make SDG can process large data efficiently.

.. code-block:: python

    """
    Example for CTGAN
    """
    from sdgx.data_connectors.csv_connector import CsvConnector
    from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
    from sdgx.synthesizer import Synthesizer
    from sdgx.utils import download_demo_data

    # This will download demo data to ./dataset
    dataset_csv = download_demo_data()

    # Create data connector for csv file
    data_connector = CsvConnector(path=dataset_csv)

    # Initialize synthesizer, use CTGAN model
    synthesizer = Synthesizer(
        model=CTGANSynthesizerModel(epochs=1),  # For quick demo
        data_connector=data_connector,
    )

    # Fit the model
    synthesizer.fit()

    # Sample
    sampled_data = synthesizer.sample(1000)
    print(sampled_data)

Save and load :ref:`Synthesizer <Synthesizer>` for future use
---------------------------------------------------------------------------------

SDG use cloudpickle to save and load :ref:`Synthesizer <Synthesizer>` for future use, which
is a powerful pickler and makes it possible to serialize Python constructs not supported by the default pickle module from the Python standard library.

Cloudpickle is especially useful for cluster computing where Python code is shipped over the network to execute on remote hosts, possibly close to the data.
And you can learn more about cloudpickle from `their repo <https://github.com/cloudpipe/cloudpickle>`_ .


.. code-block:: python

    from pathlib import Path
    import time
    _HERE = Path(__file__).parent
    date = time.strftime("%Y%m%d-%H%M%S")
    save_dir = _HERE / f"./ctgan-{date}-model"

    # Save fitted model
    synthesizer.save(save_dir)

    # Load model, then sample
    synthesizer = Synthesizer.load(save_dir, model=CTGANSynthesizerModel)
    sampled_data = synthesizer.sample(1000)
    print(sampled_data)


Evaluation
---------------------------------------------------------------------------------


.. TODO: Evaluation has not been fully implemented yet.

.. code-block:: python

    from sdgx.metrics.column.jsd import JSD

    JSD = JSD()


    selected_columns = ["workclass"]
    isDiscrete = True
    metrics = JSD.calculate(data_connector.read(), sampled_data, selected_columns, isDiscrete)

    print("JSD metric of column %s: %g" % (selected_columns[0], metrics))



Next Step
---------------------------------------------------------------------------------

- :ref:`Synthetic single-table data <Synthetic single-table data>`
- :ref:`Synthetic multi-table data <Synthetic multi-table data>`
- :ref:`Evaluation synthetic data <Evaluation synthetic data>`
