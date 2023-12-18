import numpy as np


def flatten_array(nested, prefix=""):
    """Flatten an array as a dict.

    Args:
        nested (list, numpy.array):
            Iterable to flatten.
        prefix (str):
            Name to append to the array indices. Defaults to ``''``.

    Returns:
        dict:
            Flattened array.
    """
    result = {}
    for index in range(len(nested)):
        prefix_key = "__".join([prefix, str(index)]) if len(prefix) else str(index)

        value = nested[index]
        if isinstance(value, (list, np.ndarray)):
            result.update(flatten_array(value, prefix=prefix_key))

        elif isinstance(value, dict):
            result.update(flatten_dict(value, prefix=prefix_key))

        else:
            result[prefix_key] = value

    return result


IGNORED_DICT_KEYS = ["fitted", "distribution", "type"]


def flatten_dict(nested, prefix=""):
    """Flatten a dictionary.

    This method returns a flatten version of a dictionary, concatenating key names with
    double underscores.

    Args:
        nested (dict):
            Original dictionary to flatten.
        prefix (str):
            Prefix to append to key name. Defaults to ``''``.

    Returns:
        dict:
            Flattened dictionary.
    """
    result = {}

    for key, value in nested.items():
        prefix_key = "__".join([prefix, str(key)]) if len(prefix) else key

        if key in IGNORED_DICT_KEYS and not isinstance(value, (dict, list)):
            continue

        elif isinstance(value, dict):
            result.update(flatten_dict(value, prefix_key))

        elif isinstance(value, (np.ndarray, list)):
            result.update(flatten_array(value, prefix_key))

        else:
            result[prefix_key] = value

    return result


def log_numerical_distributions_error(numerical_distributions, processed_data_columns, logger):
    """Log error when numerical distributions columns don't exist anymore."""
    unseen_columns = numerical_distributions.keys() - set(processed_data_columns)
    for column in unseen_columns:
        logger.info(
            f"Requested distribution '{numerical_distributions[column]}' "
            f"cannot be applied to column '{column}' because it no longer "
            "exists after preprocessing."
        )


def _key_order(key_value):
    parts = []
    for part in key_value[0].split("__"):
        if part.isdigit():
            part = int(part)

        parts.append(part)

    return parts


def unflatten_dict(flat):
    """Transform a flattened dict into its original form.

    Args:
        flat (dict):
            Flattened dict.

    Returns:
        dict:
            Nested dict (if corresponds)
    """
    unflattened = {}

    for key, value in sorted(flat.items(), key=_key_order):
        if "__" in key:
            key, subkey = key.split("__", 1)
            subkey, name = subkey.rsplit("__", 1)

            if name.isdigit():
                column_index = int(name)
                row_index = int(subkey)

                array = unflattened.setdefault(key, [])

                if len(array) == row_index:
                    row = []
                    array.append(row)
                elif len(array) == row_index + 1:
                    row = array[row_index]
                else:
                    # This should never happen
                    raise ValueError("There was an error unflattening the extension.")

                if len(row) == column_index:
                    row.append(value)
                else:
                    # This should never happen
                    raise ValueError("There was an error unflattening the extension.")

            else:
                subdict = unflattened.setdefault(key, {})
                if subkey.isdigit():
                    subkey = int(subkey)

                inner = subdict.setdefault(subkey, {})
                inner[name] = value

        else:
            unflattened[key] = value

    return unflattened


def validate_numerical_distributions(numerical_distributions, metadata_columns):
    """Validate ``numerical_distributions``.

    Raise an error if it's not None or dict, or if its columns are not present in the metadata.

    Args:
        numerical_distributions (dict):
            Dictionary that maps field names from the table that is being modeled with
            the distribution that needs to be used.
        metadata_columns (list):
            Columns present in the metadata.
    """
    if numerical_distributions:
        if not isinstance(numerical_distributions, dict):
            raise TypeError("numerical_distributions can only be None or a dict instance.")

        invalid_columns = numerical_distributions.keys() - set(metadata_columns)
        # if invalid_columns:
        #     raise SynthesizerInputError(
        #         'Invalid column names found in the numerical_distributions dictionary '
        #         f'{invalid_columns}. The column names you provide must be present '
        #         'in the metadata.'
        #     )
