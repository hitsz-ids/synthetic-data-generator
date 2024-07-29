from __future__ import annotations

import re
from typing import Any

import pandas as pd

from sdgx.data_models.inspectors.base import Inspector
from sdgx.exceptions import InspectorInitError


class RegexInspector(Inspector):
    """RegexInspector
    RegexInspector is a sdgx inspector that uses regular expression rules to detect column data types. It can be initialized with a custom expression, or it can be inherited and applied to specific data types,such as email, US address, HKID etc.

    By default, we will not directly register the RegexInspector to the Inspector Manager. Instead, use it as a baseclass or user-defined regex, then put it into the Inspector Manager or use it alone
    """

    pattern: str = None
    """
    pattern is the regular expression string of current inspector.
    """

    data_type_name: str = None
    """
    data_type_name is the name of the data type, such as email, US address, HKID etc.
    """

    _match_percentage: float = 0.8
    """
    Private variable used to store property match_percentage's value.
    """

    @property
    def match_percentage(self):
        """
        The match_percentage shoud > 0.5 and < 1.

        Due to the existence of empty data, wrong data, etc., the match_percentage is the proportion of the current regular expression compound. When the number of compound regular expressions is higher than this ratio, the column can be considered fit the current data type.
        """
        return self._match_percentage

    @match_percentage.setter
    def match_percentage(self, value):
        if value > 0.5 and value <= 1:
            self._match_percentage = value
        else:
            raise InspectorInitError("The match_percentage should be set in (0.5, 1].")

    def __init__(
        self,
        pattern: str = None,
        data_type_name: str = None,
        match_percentage: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.regex_columns: set[str] = set()

        # this pattern should be a re pattern
        if pattern:
            self.pattern = pattern
        # check pattern
        if self.pattern is None:
            raise InspectorInitError("Regular expression NOT found.")
        self.p = re.compile(self.pattern)

        # set data_type_name
        if data_type_name:
            if data_type_name.endswith("_columns"):
                self.data_type_name = data_type_name[:-8]
            else:
                self.data_type_name = data_type_name
        elif not self.data_type_name:
            self.data_type_name = f"regex_{self.pattern}_columns"
        # then chech the data type name
        if self.data_type_name is None:
            raise InspectorInitError("Inspector's data type undefined.")

        # set percentage
        if match_percentage:
            self.match_percentage = match_percentage

    def fit(self, input_raw_data: pd.DataFrame, *args, **kwargs):
        """Fit the inspector.

        Finds the list of regex columns from the tabular data (in pd.DataFrame).

        Args:
            raw_data (pd.DataFrame): Raw data
        """
        for each_col in input_raw_data.columns:
            each_match_rate = self._fit_column(input_raw_data[each_col])
            if each_match_rate > self.match_percentage:
                self.regex_columns.add(each_col)

        self.ready = True

    def domain_verification(self, each_sample: str):
        """
        The function domain_verification is used to add custom domain verification logic. When a sample matches a regular expression, the domain_verification function is executed for further verification.

        Additional logic checks can be performed beyond regular expressions, making it more flexible. For example, in a company name, there may be address information. When determining the type of address, if the sample ends with "Company", domain_verification can return False to avoid misclassification, thus improving the accuracy of the inspector.

        This function has the power to veto. When the function outputs False, the sample will be classified as not matching the corresponding data type of the inspector.

        If this function is not overwritten, domain_verification will default to return True.

        Args:
            each_sample (str): string of each sample.
        """

        return True

    def _fit_column(self, column_data: pd.Series):
        """
        Regular expression matching for a single column, returning the matching ratio.

        Args:
             column_data (pd.Series): the column data.
        """
        length = len(column_data)
        unmatch_cnt = 0
        match_cnt = 0
        for i in column_data:
            m = re.match(self.p, str(i))
            d = self.domain_verification(str(i))
            if m and d:
                match_cnt += 1
            else:
                unmatch_cnt += 1
                if unmatch_cnt > length * (1 - self.match_percentage) + 1:
                    break
        return match_cnt / length

    def inspect(self, *args, **kwargs) -> dict[str, Any]:
        """Inspect raw data and generate metadata."""

        return {self.data_type_name + "_columns": list(self.regex_columns)}
