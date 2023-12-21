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

- :ref:`Cacher for DataLoader <api_reference/cachers-extension>`
- :ref:`Data Connector <api_reference/data-connectors-extension>`
- :ref:`Data Processor <api_reference/data-processors-extension>`
- :ref:`Inspector for Metadata <api_reference/data-models-inspectors-extension>`
- :ref:`Model <api_reference/models-extension>`
- :ref:`Data Exporter <api_reference/data-exporters-extension>`
