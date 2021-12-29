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
from sklearn.neighbors import KNeighborsRegressor
from ncore.models.baselines.base_model import PickleableBaseModel, HyperparamMixin
from ncore.models.baselines.concatenate_composite_model import ConcatenateCompositeModel


class KNearestNeighbours(ConcatenateCompositeModel, PickleableBaseModel, HyperparamMixin):
    def __init__(self, num_treatments=None, output_dim=None, missing_treatment_resolution_strategy="nearest"):
        super(KNearestNeighbours, self).__init__(
            num_treatments, output_dim, missing_treatment_resolution_strategy
        )

    @staticmethod
    def get_hyperparameter_ranges():
        ranges = {

        }
        return ranges

    def _build_model(self, x_train, m_train, h_train, t_train, s_train, y_train):
        return KNeighborsRegressor(n_neighbors=1)
