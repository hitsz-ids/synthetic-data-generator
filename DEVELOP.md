Install pre-commit before commit

```
pip install pre-commit
pre-commit install
```

Pre-commit will automatically format the code before each commit, It can also be executed manually on all files

```
pre-commit run --all-files
```

Install package locally

```
pip install -e .[test]
```

Run unit-test before PR, **ensure that new features are covered by unit tests**

```
pytest -v
```
