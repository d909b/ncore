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
import os
import json
import numpy as np
from ncore.models.baselines.base_model import BaseModel
from ncore.models.baselines.base_model import HyperparamMixin
from ncore.models.baselines.ganite_base.ganite_model import GANITEModel
from ncore.models.baselines.base_neural_network import BaseNeuralNetwork
from ncore.models.baselines.tarnet_base.model_factory import ModelFactory


class GANITE(BaseNeuralNetwork, HyperparamMixin):
    def __init__(self, num_treatments=0, early_stopping_patience=13, best_model_path="", batch_size=32, num_epochs=100,
                 input_shape=(1,), output_dim=1, p_dropout=0.0, l2_weight=0.0, learning_rate=0.001, num_units=128,
                 num_layers=2, with_bn=False, verbose=2, ganite_weight_alpha=1.0,
                 ganite_weight_beta=1.0):
        super(GANITE, self).__init__(num_treatments, early_stopping_patience, best_model_path, batch_size, num_epochs,
                                     input_shape, output_dim, p_dropout, l2_weight, learning_rate, num_units,
                                     num_layers, with_bn, verbose)
        self.ganite_weight_alpha = ganite_weight_alpha
        self.ganite_weight_beta = ganite_weight_beta

    @staticmethod
    def get_hyperparameter_ranges():
        ranges = BaseNeuralNetwork.get_hyperparameter_ranges()
        return ranges

    def fit(self, x_train, m_train, h_train, t_train, s_train, y_train,
            x_val, m_val, h_val, t_val, s_val, y_val):
        if self.model is None:
            self.model = self._build()

        concatenated_covariates = np.concatenate([x_train, m_train, h_train], axis=-1)
        concatenated_covariates_val = np.concatenate([x_val, m_val, h_val], axis=-1)

        t_train = np.argmax(t_train, axis=-1)
        t_val = np.argmax(t_val, axis=-1)

        def random_cycle_generator(x, t, y):
            num_samples = len(x)
            num_steps = int(np.ceil(num_samples / float(self.batch_size)))

            def inner_generator():
                while True:
                    samples = np.random.permutation(len(x))
                    for _ in range(num_steps):
                        batch_indices = samples[:self.batch_size]
                        samples = samples[self.batch_size:]

                        batch_x = np.array([x[idx] for idx in batch_indices])
                        batch_t = np.array([t[idx] for idx in batch_indices])
                        batch_y = y[batch_indices]

                        yield (batch_x, batch_t), batch_y

            return inner_generator(), num_steps

        train_generator, train_steps = random_cycle_generator(concatenated_covariates, t_train, y_train)
        val_generator, val_steps = random_cycle_generator(concatenated_covariates_val, t_val, y_val)

        self.model.train(
            train_generator, train_steps,
            val_generator, val_steps,
            self.num_epochs, self.learning_rate, learning_rate_decay=0.97, iterations_per_decay=100,
            dropout=self.p_dropout, imbalance_loss_weight=0.0, l2_weight=self.l2_weight,
            checkpoint_path=self.best_model_path, early_stopping_patience=self.early_stopping_patience,
            early_stopping_on_pehe=False
        )

    def _build(self):
        return GANITEModel(
            input_dim=self.input_shape[-1],
            output_dim=self.output_dim,
            num_units=self.num_units,
            num_layers=self.num_layers,
            dropout=self.p_dropout,
            l2_weight=self.l2_weight,
            learning_rate=self.learning_rate,
            num_treatments=2**self.num_treatments,
            nonlinearity="elu",
            with_bn=self.with_bn,
            alpha=self.ganite_weight_alpha,
            beta=self.ganite_weight_beta
        )

    @staticmethod
    def load(save_folder_path):
        config_file_name = BaseNeuralNetwork.get_config_file_name()
        config_file_path = os.path.join(save_folder_path, config_file_name)
        with open(config_file_path, "r") as fp:
            config = json.load(fp)

        input_shape = config["input_shape"]
        output_dim = config["output_dim"]
        l2_weight = config["l2_weight"]
        num_units = config["num_units"]
        learning_rate = config["learning_rate"]
        p_dropout = config["p_dropout"]
        num_layers = config["num_layers"]
        with_bn = config["with_bn"]
        num_epochs = config["num_epochs"]
        early_stopping_patience = config["early_stopping_patience"]
        verbose = config["verbose"]
        batch_size = config["batch_size"]
        num_treatments = config["num_treatments"]
        best_model_path = config["best_model_path"]
        ganite_weight_alpha = config["ganite_weight_alpha"]
        ganite_weight_beta = config["ganite_weight_beta"]

        instance = GANITE(
            input_shape=input_shape,
            output_dim=output_dim,
            l2_weight=l2_weight,
            num_units=num_units,
            learning_rate=learning_rate,
            p_dropout=p_dropout,
            num_layers=num_layers,
            with_bn=with_bn,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            verbose=verbose,
            batch_size=batch_size,
            best_model_path=best_model_path,
            num_treatments=num_treatments,
            ganite_weight_alpha=ganite_weight_alpha,
            ganite_weight_beta=ganite_weight_beta
        )

        weight_list = ModelFactory.load_weights(os.path.join(best_model_path, GANITE.get_save_file_name()))
        instance.model = instance._build()
        instance.model.set_weights(weight_list)
        return instance

    def get_config(self):
        base_config = super(GANITE, self).get_config()
        config = {
            "ganite_weight_alpha": self.ganite_weight_alpha,
            "ganite_weight_beta": self.ganite_weight_beta,
        }
        base_config.update(config)
        return base_config

    def save(self, save_folder_path, overwrite=True):
        BaseModel.save_config(save_folder_path, self.get_config(), BaseNeuralNetwork.get_config_file_name(), overwrite,
                              GANITE)
        ModelFactory.save_weights(self.model, os.path.join(save_folder_path, GANITE.get_save_file_name()))
