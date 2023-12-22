Command Line Interface
==================================================

Command Line Interface(CLI) is designed to simplify the usage of SDG and enable other programs to use SDG in a more convenient way.

There are tow main commands in the CLI:

- ``fit``: For fitting, finetuning, retraining... the model, which will save the final model to a specified path.
- ``sample``: Load existing model and sample synthetic data.

And as SDG supports plug-in system, users can list all available via ``list-{component}`` command.

.. Note::

    If you want to use SDG as a library, please refer to :ref:`Use Synthetic Data Generator as a library <Use Synthetic Data Generator as a library>`.

    If you want to extend SDG with your own components, please refer to :ref:`Developer guides for extension <Extented Synthetic Data Generator>`.

CLI for synthetic single-table data
--------------------------------------------------

.. click:: sdgx.cli.main:cli
   :prog: sdgx
   :nested: full
