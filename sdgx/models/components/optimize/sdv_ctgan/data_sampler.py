"""DataSampler module."""

from __future__ import annotations

from typing import List

import numpy as np
from tqdm import autonotebook as tqdm

from sdgx.models.components.optimize.ndarray_loader import NDArrayLoader
from sdgx.models.components.optimize.sdv_ctgan.types import SpanInfo


class DataSampler(object):
    """DataSampler samples the conditional vector and corresponding data for CTGAN."""

    def __init__(self, data: NDArrayLoader | np.ndarray, output_info: List[List[SpanInfo]], log_frequency: bool):
        self._data: NDArrayLoader | np.ndarray = data

        def is_onehot_encoding_column(column_info: List[SpanInfo]):
            # Notice: Because of historical reasons, this is related to `_fit_discrete` in `DataTransformer`.
            return len(column_info) == 1 and column_info[0].activation_fn == "softmax"

        n_onehot_columns = sum(
            [1 for column_info in output_info if is_onehot_encoding_column(column_info)]
        )

        self._onehot_column_matrix_st = np.zeros(n_onehot_columns, dtype="int32")

        # Store the row id for each category in each discrete column.
        # For example _rid_by_cat_cols[a][b] is a list of all rows with the
        # a-th discrete column equal value b.
        self._rid_by_cat_cols = []

        # Compute _rid_by_cat_cols
        st = 0
        for column_info in output_info:
            if is_onehot_encoding_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim

                rid_by_cat = []
                for j in range(span_info.dim):
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0])
                self._rid_by_cat_cols.append(rid_by_cat)
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])
        assert st == data.shape[1]

        # Prepare an interval matrix for efficiently sample conditional vector
        max_category = max(
            [
                column_info[0].dim
                for column_info in output_info
                if is_onehot_encoding_column(column_info)
            ],
            default=0,
        )

        self._onehot_column_cond_st = np.zeros(n_onehot_columns, dtype="int32")
        self._onehot_column_n_category = np.zeros(n_onehot_columns, dtype="int32")
        self._onehot_column_category_prob = np.zeros((n_onehot_columns, max_category))
        self._n_onehot_columns = n_onehot_columns
        self._n_categories = sum(
            [
                column_info[0].dim
                for column_info in output_info
                if is_onehot_encoding_column(column_info)
            ]
        )

        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if is_onehot_encoding_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim
                category_freq = np.sum(data[:, st:ed], axis=0)
                if log_frequency:
                    category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)
                self._onehot_column_category_prob[current_id, : span_info.dim] = category_prob
                self._onehot_column_cond_st[current_id] = current_cond_st
                self._onehot_column_n_category[current_id] = span_info.dim
                current_cond_st += span_info.dim
                current_id += 1
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])
        assert st == data.shape[1]

    def _random_choice_prob_index(self, discrete_column_id):
        probs = self._onehot_column_category_prob[discrete_column_id]
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
        return (probs.cumsum(axis=1) > r).argmax(axis=1)

    def sample_condvec(self, batch):
        """Generate the conditional vector for training.

        Returns:
            cond (batch x #categories):
                The conditional vector.
            mask (batch x #discrete columns):
                A one-hot vector indicating the selected discrete column.
            discrete column id (batch):
                Integer representation of mask.
            category_id_in_col (batch):
                Selected category in the selected discrete column.
        """
        if self._n_onehot_columns == 0:
            return None

        onehot_column_id = np.random.choice(np.arange(self._n_onehot_columns), batch)

        cond = np.zeros((batch, self._n_categories), dtype="float32")
        mask = np.zeros((batch, self._n_onehot_columns), dtype="float32")
        mask[np.arange(batch), onehot_column_id] = 1
        category_id_in_col = self._random_choice_prob_index(onehot_column_id)
        category_id = self._onehot_column_cond_st[onehot_column_id] + category_id_in_col
        cond[np.arange(batch), category_id] = 1

        return cond, mask, onehot_column_id, category_id_in_col

    def sample_original_condvec(self, batch):
        """Generate the conditional vector for generation use original frequency."""
        if self._n_onehot_columns == 0:
            return None

        cond = np.zeros((batch, self._n_categories), dtype="float32")

        for i in tqdm.tqdm(range(batch), desc="Sampling in batch", delay=3, leave=False):
            row_idx = np.random.randint(0, len(self._data))
            col_idx = np.random.randint(0, self._n_onehot_columns)
            matrix_st = self._onehot_column_matrix_st[col_idx]
            matrix_ed = matrix_st + self._onehot_column_n_category[col_idx]
            pick = np.argmax(self._data[row_idx, matrix_st:matrix_ed])
            cond[i, pick + self._onehot_column_cond_st[col_idx]] = 1

        return cond

    def sample_data(self, n, col, opt):
        """Sample data from original training data satisfying the sampled conditional vector.

        Returns:
            n rows of matrix data.
        """
        if col is None:
            idx = np.random.randint(len(self._data), size=n)
            return self._data[idx]

        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self._rid_by_cat_cols[c][o]))

        return self._data[idx]

    def dim_cond_vec(self):
        """Return the total number of categories."""
        return self._n_categories

    def generate_cond_from_condition_column_info(self, condition_info, batch):
        """Generate the condition vector."""
        vec = np.zeros((batch, self._n_categories), dtype="float32")
        id_ = self._onehot_column_matrix_st[condition_info["discrete_column_id"]]
        id_ += condition_info["value_id"]
        vec[:, id_] = 1
        return vec
