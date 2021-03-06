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
from tensorflow.keras.utils import to_categorical
from ncore.data_access.preprocessing.base_processor import BasePreprocessor
from ncore.data_access.meta_data.feature_types import FeatureTypeContinuous, FeatureTypeDiscrete


class OneHotDiscretise(BasePreprocessor):
    def __init__(self, max_num_elements_for_discretisation=6):
        super(OneHotDiscretise, self).__init__()
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
            if unique_values == [0.0]:  # Include both binary values, even if not included in the training set.
                unique_values.append(1.0)
            num_unique = len(unique_values)
            one_hot_transform = num_unique <= self.max_num_elements_for_discretisation

            self.col_state.append((unique_values, num_unique, one_hot_transform))

    def transform(self, x, y=None, transform_fun_name=None):
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=-1)
        num_cols = len(self.col_state)
        with_feature_names = self.feature_names is not None

        new_x, new_feature_names, new_feature_types = [], [], []
        for j in range(num_cols):
            unique_values, num_unique, one_hot_transform = self.col_state[j]

            if one_hot_transform:
                # NOTE: "nan"s become a separate class.
                index_map = dict(zip(unique_values, range(num_unique)))
                safe_values = list(map(lambda xi: "nan" if isfloat(xi) and (np.isnan(float(xi)) or np.isinf(xi)) else
                                                  xi, x[:, j]))
                converted = np.array(list(map(lambda xi: index_map[xi] if xi in index_map else
                                                         index_map["nan"] if "nan" in index_map else index_map["unknown"],
                                              safe_values)))

                feature_type = FeatureTypeDiscrete(num_unique)
                if num_unique > 2:
                    converted = to_categorical(converted, num_classes=num_unique)

                    if with_feature_names:
                        reverse_index_map = {v: k for k, v in index_map.items()}
                        for class_idx in range(num_unique):
                            new_feature_names.append(self.feature_names[j] + "_" + str(reverse_index_map[class_idx]))
                    for _ in range(num_unique):
                        new_feature_types.append(feature_type)  # Re-use same instance.
                else:
                    converted = np.expand_dims(converted, axis=-1)
                    if with_feature_names:
                        new_feature_names.append(self.feature_names[j])
                    new_feature_types.append(feature_type)
            else:
                # Do not change continuous variables.
                converted = np.expand_dims(x[:, j], axis=-1)
                if with_feature_names:
                    new_feature_names.append(self.feature_names[j])
                new_feature_types.append(FeatureTypeContinuous())
            new_x.append(converted)
        new_x = np.column_stack(new_x)
        self.feature_types = new_feature_types
        assert len(self.feature_types) == new_x.shape[-1]
        if with_feature_names:
            self.feature_names = new_feature_names
            assert len(self.feature_names) == new_x.shape[-1]
        return new_x
