# Development Guide

## Code Style and Lint

We use [black](https://github.com/psf/black) as the code formatter, the best way to use it is to install the pre-commit hook, it will automatically format the code before each commit

Install pre-commit before commit

```bash
pip install pre-commit
pre-commit install
```

Pre-commit will automatically format the code before each commit, It can also be executed manually on all files

```bash
pre-commit run --all-files
```

Comment style follows [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

## Install Locally with Test Dependencies

```bash
pip install -e .[test]
```

## Unit tests

We use pytest to write unit tests, and use pytest-cov to generate coverage reports

```bash
pytest -v
pytest --cov=sdgx # Generate coverage reports
```

Run unit-test before PR, **ensure that new features are covered by unit tests**

## Build Docs

Install docs dependencies

```bash
pip install -e .[docs]
```

Build docs

```bash
cd docs && make html
```

Use [start-docs-host.sh](dev-tools/start-docs-host.sh) to deploy a local http server to view the docs

```bash
cd ./dev-tools && ./start-docs-host.sh
```

Access `http://localhost:8080` for docs.
