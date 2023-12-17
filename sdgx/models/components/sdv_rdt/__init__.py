# -*- coding: utf-8 -*-

"""Top-level package for RDT."""


__author__ = """MIT Data To AI Lab"""
__email__ = "dailabmit@gmail.com"
__version__ = "1.2.1"

import numpy as np
import pandas as pd

from sdgx.models.components.sdv_rdt import transformers
from sdgx.models.components.sdv_rdt.hyper_transformer import HyperTransformer

__all__ = ["HyperTransformer", "transformers"]

RANDOM_SEED = 42


def get_demo(num_rows=5):
    """Generate demo data with multiple sdtypes.

    The first five rows are hard coded. The rest are randomly generated
    using ``np.random.seed(42)``.

    Args:
        num_rows (int):
            Number of data rows to generate. Defaults to 5.

    Returns:
        pd.DataFrame
    """
    # Hard code first five rows
    login_dates = ["2021-06-26", "2021-02-10", "NAT", "2020-09-26", "2020-12-22"]
    last_login = [np.datetime64(i) for i in login_dates]
    email_optin = pd.Series([False, False, False, True, np.nan], dtype="object")
    credit_card = ["VISA", "VISA", "AMEX", np.nan, "DISCOVER"]
    age = [29, 18, 21, 45, 32]
    dollars_spent = [99.99, np.nan, 2.50, 25.00, 19.99]

    data = pd.DataFrame(
        {
            "last_login": last_login,
            "email_optin": email_optin,
            "credit_card": credit_card,
            "age": age,
            "dollars_spent": dollars_spent,
        }
    )

    if num_rows <= 5:
        return data.iloc[:num_rows]

    # Randomly generate the remaining rows
    random_state = np.random.get_state()
    np.random.set_state(np.random.RandomState(RANDOM_SEED).get_state())
    try:
        num_rows -= 5

        login_dates = np.array(
            [
                np.datetime64("2000-01-01") + np.timedelta64(np.random.randint(0, 10000), "D")
                for _ in range(num_rows)
            ]
        )
        login_dates[np.random.random(size=num_rows) > 0.8] = np.datetime64("NaT")

        email_optin = pd.Series([True, False, np.nan], dtype="object").sample(
            num_rows, replace=True
        )
        credit_card = np.random.choice(["VISA", "AMEX", np.nan, "DISCOVER"], size=num_rows)
        age = np.random.randint(18, 100, size=num_rows)

        dollars_spent = np.around(np.random.uniform(0, 100, size=num_rows), decimals=2)
        dollars_spent[np.random.random(size=num_rows) > 0.8] = np.nan

    finally:
        np.random.set_state(random_state)

    return data.append(
        pd.DataFrame(
            {
                "last_login": login_dates,
                "email_optin": email_optin,
                "credit_card": credit_card,
                "age": age,
                "dollars_spent": dollars_spent,
            }
        ),
        ignore_index=True,
    )
