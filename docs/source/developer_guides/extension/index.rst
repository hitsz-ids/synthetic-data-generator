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

- :ref:`Cacher for DataLoader <api_reference/cacher-extension>`
- :ref:`Data Connector <_api_reference/data-connector-extension>`
- :ref:`Data Processor <_api_reference/data-processor-extension>`
- :ref:`Model <_api_reference/model-extension>`
