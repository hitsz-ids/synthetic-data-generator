# Making a new release of duetector

## Manual release

### Python package

This project can be distributed as Python
packages. Before generating a package, we first need to install `build`.

```bash
pip install build twine hatch
```

Bump the version using `hatch`.

```bash
hatch version <new-version>
```

To create a Python source package (`.tar.gz`) and the binary package (`.whl`) in the `dist/` directory, do:

```bash
rm -rf dist/*
python -m build
```

> `python setup.py sdist bdist_wheel` is deprecated and will not work for this package.

Then to upload the package to PyPI, do:

```bash
twine upload dist/*
```

### Python package and Docker image

The version number needs to be changed manually before proceeding with the release.

```bash
hatch version <new-version>
```

Once there is a release, [Github Action](https://github.com/hitsz-ids/duetector/actions/workflows/publish.yml) will automatically publish python package.
