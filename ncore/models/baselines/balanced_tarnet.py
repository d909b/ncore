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
from ncore.models.baselines.tarnet import TARNET
from ncore.models.baselines.base_neural_network import BaseNeuralNetwork


class BalancedTARNET(TARNET):
    def __init__(self, num_treatments=0, early_stopping_patience=13, best_model_path="", batch_size=32, num_epochs=100,
                 input_shape=(1,), output_dim=1, p_dropout=0.0, l2_weight=0.0, learning_rate=0.001, num_units=128,
                 num_layers=2, with_bn=False, verbose=2):
        super(BalancedTARNET, self).__init__(
            num_treatments, early_stopping_patience, best_model_path, batch_size,
            num_epochs, input_shape, output_dim, p_dropout, l2_weight, learning_rate, num_units,
            num_layers, with_bn, verbose, imbalance_loss_weight=1.0)

    @staticmethod
    def get_hyperparameter_ranges():
        ranges = TARNET.get_hyperparameter_ranges()
        return ranges

    @staticmethod
    def get_subclass_kwargs(config):
        return {}

    def get_config(self):
        config = super(BalancedTARNET, self).get_config()
        del config["imbalance_loss_weight"]
        return config

    @staticmethod
    def load(save_folder_path, base_class=None):
        return BaseNeuralNetwork.load(save_folder_path, base_class=BalancedTARNET)
