"""
Copyright (C) 2021  Patrick Schwab, GlaxoSmithKline plc, Sonali Parbhoo, Harvard University
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
from abc import abstractmethod
from collections import defaultdict
from ncore.models.baselines.base_model import BaseModel
from ncore.apps.util import convert_int_to_indicator_list, convert_indicator_list_to_int, jaccard_distance


class CompositeModel(BaseModel):
    def __init__(self, num_treatments=None, output_dim=None, missing_treatment_resolution_strategy="nearest"):
        super(CompositeModel, self).__init__(num_treatments)
        self.model = None
        self.output_dim = output_dim
        self.missing_treatment_resolution_strategy = missing_treatment_resolution_strategy

    @abstractmethod
    def _build_model(self, x_train, m_train, h_train, t_train, s_train, y_train):
        """
        Called for a specific treatment t = T. Overwritten in child classes.
        """
        raise NotImplementedError()

    @abstractmethod
    def _predict_for_model(self, model, x, m, h, t, s):
        """
        Called for a specific treatment t = T. Overwritten in child classes.
        """
        raise NotImplementedError()

    @abstractmethod
    def _fit_for_model(self, model, x_train, m_train, h_train, t_train, s_train, y_train,
                      x_val, m_val, h_val, t_val, s_val, y_val):
        """
        Called for a specific treatment t = T. Overwritten in child classes.
        """
        raise NotImplementedError()

    @staticmethod
    def get_x_by_idx(data, indices):
        if isinstance(data, np.ndarray):
            return data[indices]
        else:
            return np.array([data[idx] for idx in indices])

    def resolve_missing_model_nearest(self, indicators):
        min_dist = float("inf")
        closest = -1
        for treatment_offset in sorted(self.model.keys()):
            this_indicators = convert_int_to_indicator_list(treatment_offset,
                                                            min_length=self.num_treatments)
            dist = jaccard_distance(indicators, this_indicators)
            if dist < min_dist:
                min_dist = dist
                closest = treatment_offset
        return self.model[closest]

    def resolve_missing_model(self, indicators):
        if self.missing_treatment_resolution_strategy == "nearest":
            return self.resolve_missing_model_nearest(indicators)

    def predict(self, x, m, h, t, s):
        results = np.zeros((len(t), self.output_dim))
        treatment_split = CompositeModel.get_treatment_assignments(t)
        for treatment_offset in sorted(treatment_split.keys()):  # Enumerate all possible combinations.
            indicators = convert_int_to_indicator_list(treatment_offset, min_length=self.num_treatments)
            indices = treatment_split[treatment_offset]
            current_covariates = [CompositeModel.get_x_by_idx(field, indices) for field in [x, m, h, t, s]]

            if treatment_offset not in self.model:
                # Use another model if no combination-specific model is available.
                selected_model = self.resolve_missing_model(indicators)
            else:
                selected_model = self.model[treatment_offset]
            y_pred = self._predict_for_model(selected_model, *current_covariates)
            if len(y_pred.shape) == 1:
                y_pred = np.expand_dims(y_pred, axis=-1)
            results[indices] = y_pred
        return results

    @staticmethod
    def get_treatment_assignments(t_values):
        treatment_split = defaultdict(list)
        for idx, this_t in enumerate(t_values):
            treatment_split[convert_indicator_list_to_int(this_t)].append(idx)
        return treatment_split

    def fit(self, x_train, m_train, h_train, t_train, s_train, y_train,
            x_val, m_val, h_val, t_val, s_val, y_val):
        if self.model is None:
            self.model = {}

        treatment_split_train = CompositeModel.get_treatment_assignments(t_train)
        treatment_split_val = CompositeModel.get_treatment_assignments(t_val)

        # Enumerate all observed combinations (Note: May not have observed all possible combinations)
        for treatment_offset in treatment_split_train.keys():
            indices_train = treatment_split_train[treatment_offset]
            indices_val = treatment_split_val[treatment_offset]
            train_inputs = [CompositeModel.get_x_by_idx(field, indices_train) for field in
                            [x_train, m_train, h_train, t_train, s_train, y_train]]
            val_inputs = [CompositeModel.get_x_by_idx(field, indices_val) for field in
                          [x_val, m_val, h_val, t_val, s_val, y_val]]
            self.model[treatment_offset] = self._build_model(x_train, m_train, h_train, t_train, s_train, y_train)
            self._fit_for_model(self.model[treatment_offset], *train_inputs, *val_inputs)
