Architecture
====================

.. NOTE::

    For our motivation, please refer to :ref:`Our motivation <Our motivation of this project>`.

Principles and Goals
-------------------------------------

The architecture designed with some principles:

- Explicit is better than implicit.
- Flat is better than nested.
- Favor object composition over class inheritance.
- Dependency inversion through interfaces and data objects.

We have the following goals:

- High performance
- Native batch-based streaming
- Scalable for large datasets
- Easy to expand


Overview
-------------------------------------


.. image:: /_static/architecture.png
    :align: center

The SDG project was designed to be easily extensible in order to allow for
leveraging the strengths of the community for project development. We call it **the SDG Ecosystem**.

In **the SDG Ecosystem**, the SDG's main focus will be on the development of the ``SDG Library``,
``Plugin System`` and ``Synthesizer``.
It's encouraged to develop the ``Data Connector`` and ``Evaluator`` according to your needs.
We'll provide some of the native support, as well as some plugin projects maintained by hitsz-ids.


Key components
-------------------------------------

- **Data Connector**: Used to connect to different data sources.
  Because data varies from organisation to organisation and mission to mission,
  users may need to develop their own Data Connector to suit their needs.
- **DataLoader**: DataLoader enables data to be loaded into memory in batches.
  To avoid network overhead from repeated reads,
  the DataLoader **SHOULD** support some caching policies.
- **Data Processor**: Data Processor will be used for pre-processing and post-processing of data.

  - ``Inspector``: Used to extract metadata such as patterns, types, etc. from raw data.
  - ``Transformer``: Used to modify data to comply with requirements, such as masking sensitive information or discarding non-compliant data.
  - ``Formatter``: Used to format non-compliant data into compliant data, such as datetime to timestamp.
    The **same** Formatter **SHOULD** be used for training and sampling.
  - ``Sampler``: Used to sample data. For sparse or huge datasets, it is a more efficient way.
    Simple sampling may lead to loss of some information, which **MAY** need to do more processing to ensure uniformity,
    *e.g. sparse category should not be missing after sampling.*
- **Model**: Models used to generate synthetic data.

  - A model needs to support incremental training as much as possible.
    DataLoader will be the entry point for its input,
    and Model can decide whether to get all the data at once according to its own implementation.
  - The Model is stateful and it **SHOULD** record the previous output. To avoids outputting duplicate information.
  - Model **SHOULD** support save/load from/to disk.
- **Plugin System**: Used to support plugin project for SDG.
  We will use `Pluggy <https://github.com/pytest-dev/pluggy>`_ to implement a zero-invasive plug-in system for SDG,
  users can call the plug-in through the **CLI** or directly use the ``Manager``.
- **Synthesizer**: All logic will be tied together by ``Synthesizer``.


Data Model
-------------------------------------

- **Metadata**: Each field of the form data, its type, and its restrictions.
  Data Processor and Model can use it to understand the raw data
  and its constraints to get more realistic fitted data.
- **ProcessedData**: The raw data will be processed to remove those that do not meet the requirements
  and the format of the data will be converted to a common format
  on which any model can be developed based on this format assumption.

.. NOTE::

    For developers, please refer to :ref:`Developer guides for data models <Data Models Specification>`.

Interaction
-------------------------------------

.. image:: /_static/interaction.png
    :align: center
    :alt: Interaction with SDG

SDG is designed as both a CLI and a library.
In addition to SDG developers, we consider the following roles:

- **User**: Scientist, researcher, or engineer, they **use SDG's CLI or library** for their research or work.
  They also are uses of SDG's plugins and downstream developers.
- **Plugin developer**: Plugin developers **build plugins for SDG**, based on SDG's plugin system.
  This may be the **best way** to develop **models** for researchers
  or develop new **data interfaces** or **processors** for developers.
- **Downstream developer**: Downstream developers use SDG to **build their own project**, for example,
  warp SDG as a service, or intergate SDG with other open-source projects or commercial products.
  The difference between plugin and downstream developers is that
  downstream projects are **self-centric**, while plugin projects are **SDG-centric**.
