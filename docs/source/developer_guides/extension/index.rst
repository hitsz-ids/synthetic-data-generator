Extented Synthetic Data Generator
=====================================

.. NOTE::

    Understand the purpose of each component from the :ref:`architecture`.

SDG uses `pluggy <https://github.com/pytest-dev/pluggy>`_ to develop plug-in systems,
which is based on the `entry-points of Python project <https://packaging.python.org/en/latest/specifications/entry-points/#entry-points>`_.

A plugin project is made up of three parts:

- A class, inherits from the ``register_type`` of :ref:`Manager <manager>`, which contains your own logic.
- A register function, which's name is defined(decorated) by ``@hookspec``.
  and you need to implement it and use ``@hookimp`` to declare it as a registed hook.
- A ``entry-points`` in ``pyproject.toml``, which pointing to the hookimp function. The subdomain of the entry-point
  is the ``PROJECT_NAME`` you can find in :ref:`manager`.


View latest extension example on `GitHub <https://github.com/hitsz-ids/synthetic-data-generator/tree/main/example/extension>`_.


Plugin-supported modules
------------------------

- :ref:`API Reference for extended Data Connector <api_reference/data-connectors-extension>`:
  :ref:`Data Connector <Data Connector>` is used to connect to data sources.
- :ref:`API Reference for extended Cacher for DataLoader <api_reference/cachers-extension>`:
  :ref:`Cacher <Cacher>` is used for improving performance,
  reducing network overhead and support large datasets.
- :ref:`API Reference for extended Data Processor <api_reference/data-processors-extension>`:
  :ref:`Data Processor <Data Processor>` is used to pre-process and post-process data.
  It is useful for business logic.
- :ref:`API Reference for extended Inspector for Metadata <api_reference/data-models-inspectors-extension>`:
  :ref:`Inspector <Inspector>` is used to extract metadata such as patterns, types, etc. from raw data.
- :ref:`API Reference for extended Model <api_reference/models-extension>`:
  :ref:`Model <SynthesizerModel>`, the model fitted by processed data and used to generate synthetic data.
- :ref:`API Reference for extended Data Exporter <api_reference/data-exporters-extension>`:
  :ref:`Data Exporter <Data Exporter>` is used to export data to somewhere.
  Use it in CLI or library way to save your processed data or synthetic data.
