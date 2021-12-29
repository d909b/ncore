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

Copyright (C) 2020  Patrick Schwab, F. Hoffmann-La Roche Ltd
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
from ncore.models.baselines.base_model import HyperparamMixin
from ncore.models.baselines.base_neural_network import BaseNeuralNetwork
from ncore.models.baselines.tarnet_base.model_builder import ModelBuilder


class TARNET(BaseNeuralNetwork, HyperparamMixin):
    def __init__(self, num_treatments=0, early_stopping_patience=13, best_model_path="", batch_size=32, num_epochs=100,
                 input_shape=(1,), output_dim=1, p_dropout=0.0, l2_weight=0.0, learning_rate=0.001, num_units=128,
                 num_layers=2, with_bn=False, verbose=2):
        super(TARNET, self).__init__(num_treatments, early_stopping_patience, best_model_path, batch_size, num_epochs,
                                     input_shape, output_dim, p_dropout, l2_weight, learning_rate, num_units,
                                     num_layers, with_bn, verbose)

    @staticmethod
    def get_hyperparameter_ranges():
        ranges = BaseNeuralNetwork.get_hyperparameter_ranges()
        return ranges

    def _build(self):
        return ModelBuilder.build_tarnet(
            input_dim=self.input_shape[-1],
            output_dim=self.output_dim,
            num_units=self.num_units,
            num_layers=self.num_layers,
            dropout=self.p_dropout,
            l2_weight=self.l2_weight,
            learning_rate=self.learning_rate,
            num_treatments=self.num_treatments,
            with_bn=self.with_bn
        )

    @staticmethod
    def load(save_folder_path, base_class=None):
        return BaseNeuralNetwork.load(save_folder_path, base_class=TARNET)
