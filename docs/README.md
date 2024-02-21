# SDG API docs

## Online docs

Typically, our [latest API document](https://synthetic-data-generator.readthedocs.io/en/latest/) can be accessed via readthedocs.

## Build docs locally

You can build the docs on your own computer.

Step 1: Install docs dependencies

```
pip install -e .[docs]
```

Step 2: Build docs

```
cd docs && make html
```

Step 3 (Optional): Use `start-docs-host.sh` to deploy a local http server to view the docs

```
cd ./dev-tools && ./start-docs-host.sh
```

Then access http://localhost:8910 for docs.
