Architecture
====================

.. NOTE::

    For our motivation, please refer to :ref:`Our motivation <Our motivation of this project>`

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



Key components
-------------------------------------


Data Model
-------------------------------------


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
