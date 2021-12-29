"""
Copyright (C) 2021  Patrick Schwab, GlaxoSmithKline plc
Copyright (C) 2019  Patrick Schwab, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import numpy as np
from ncore.apps.util import isfloat
from sklearn.preprocessing import StandardScaler
from ncore.data_access.preprocessing.base_processor import BasePreprocessor


class Standardise(BasePreprocessor):
    def __init__(self, max_num_elements_for_discretisation=6):
        super(Standardise, self).__init__()
        self.max_num_elements_for_discretisation = max_num_elements_for_discretisation
        self.col_state = None

    def fit(self, x, y=None):
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=-1)
        num_cols = x.shape[-1]

        self.col_state = []
        for j in range(num_cols):
            # Replace nan's with "nan" strings as nan itself is not a unique item in sets.
            safe_values = list(map(lambda xi:
                                   "nan" if xi is None or (isfloat(xi) and (np.isnan(float(xi)) or np.isinf(xi)))
                                   else xi, x[:, j]))
            unique_values = sorted(set(safe_values), key=lambda a: 0 if a == "nan" else a)
            num_unique = len(unique_values)

            is_continuous = num_unique > self.max_num_elements_for_discretisation
            if is_continuous:
                scaler = StandardScaler()
                converted, _, _ = BasePreprocessor.convert_to_float(x[:, j])
                filtered = np.array(list(filter(lambda xi: not (np.isnan(xi) or np.isinf(xi)), converted)))
                if len(filtered) > 0:
                    scaler.fit(filtered.reshape(-1, 1))
                else:
                    scaler = None
            else:
                scaler = None
            self.col_state.append((unique_values, num_unique, scaler))

    def transform(self, x, y=None, transform_fun_name="transform"):
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=-1)
        num_cols = len(self.col_state)

        new_x = []
        for j in range(num_cols):
            # Replace nan's with "nan" strings as nan itself is not a unique item in sets.
            unique_values, num_unique, scaler = self.col_state[j]
            is_continuous = num_unique > self.max_num_elements_for_discretisation
            if is_continuous:
                converted, _, _ = BasePreprocessor.convert_to_float(x[:, j])
                filter_idx = np.where(list(map(lambda xi: not (np.isnan(xi) or np.isinf(xi)), converted)))[0]
                if scaler is not None:
                    converted[filter_idx] = getattr(scaler, transform_fun_name)(converted[filter_idx].reshape(-1, 1))
            else:
                converted = x[:, j]
            new_x.append(converted)
        new_x = np.column_stack(new_x)
        return new_x
