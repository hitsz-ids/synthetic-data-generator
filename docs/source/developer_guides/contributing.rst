Overview
========

In
`CONTRIBUTING <https://github.com/hitsz-ids/synthetic-data-generator/blob/main/CONTRIBUTING.md>`__,
there are some Overview diagrams, refer to them for details.


Contributing guides
==================================================

Code Style and Lint
-------------------

We use `black <https://github.com/psf/black>`__ as the code formatter,
the best way to use it is to install the pre-commit hook, it will
automatically format the code before each commit

Install pre-commit before commit

.. code:: bash

   pip install pre-commit
   pre-commit install

Pre-commit will automatically format the code before each commit, It can
also be executed manually on all files

.. code:: bash

   pre-commit run --all-files

Comment style follows `Google Python Style
Guide <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`__.

Install Locally with Test Dependencies
--------------------------------------

.. code:: bash

   pip install -e .[test]

Unit tests
----------

We use pytest to write unit tests, and use pytest-cov to generate
coverage reports

.. code:: bash

   pytest -vv --cov-config=.coveragerc --cov=sdgx/ tests # Generate coverage reports

Run unit-test before PR, **ensure that new features are covered by unit
tests**

Build Docs
----------

Install docs dependencies

.. code:: bash

   pip install -e .[docs]

Build docs

.. code:: bash

   cd docs && make html

Use `start-docs-host.sh <dev-tools/start-docs-host.sh>`__ to deploy a
local http server to view the docs

.. code:: bash

   cd ./dev-tools && ./start-docs-host.sh

Access ``http://localhost:8910`` for docs.
