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
import six
import sys
import json
import numpy as np
from ncore.apps.util import info
from abc import ABCMeta, abstractmethod
from tensorflow.keras.callbacks import EarlyStopping
from ncore.models.baselines.base_model import BaseModel
from ncore.models.baselines.tarnet_base.model_factory import ModelFactory, ModelFactoryCheckpoint


if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


@six.add_metaclass(ABCMeta)
class BaseNeuralNetwork(BaseModel):
    def __init__(self, num_treatments=0, early_stopping_patience=13, best_model_path="", batch_size=32, num_epochs=100,
                 input_shape=(1,), output_dim=1, p_dropout=0.0, l2_weight=0.0, learning_rate=0.001, num_units=128,
                 num_layers=2, with_bn=False, verbose=2):
        super(BaseNeuralNetwork, self).__init__(num_treatments)
        self.model = None
        self.verbose = verbose
        self.with_bn = with_bn
        self.l2_weight = l2_weight
        self.p_dropout = p_dropout
        self.num_units = num_units
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.best_model_path = best_model_path
        self.early_stopping_patience = early_stopping_patience

    @staticmethod
    def get_hyperparameter_ranges():
        ranges = {
            "learning_rate": (0.03, 0.003),
            # "activation": ("relu", "elu", "selu"),
            "dropout": (0.0, 0.1, 0.25, 0.35),
            "num_layers": (1, 2, 3),
            "l2_weight": (0.0, 0.0001, 0.00001),
            "batch_size": (64, 128, 256, 512),
            "num_units": (32, 64, 128, 256),
        }
        return ranges

    def predict(self, x, m, h, t, s):
        if self.model is None:
            self.model = self._build()

        concatenated_covariates = np.concatenate([x, m, h], axis=-1)
        y_pred = self.model.predict([concatenated_covariates, np.argmax(t, axis=-1)])
        return y_pred

    def fit(self, x_train, m_train, h_train, t_train, s_train, y_train,
            x_val, m_val, h_val, t_val, s_val, y_val):
        if self.model is None:
            self.model = self._build()

        # Save once up front in case training does not converge.
        ModelFactory.save_weights(
            self.model, os.path.join(self.best_model_path, BaseNeuralNetwork.get_save_file_name())
        )

        concatenated_covariates = np.concatenate([x_train, m_train, h_train], axis=-1)
        concatenated_covariates_val = np.concatenate([x_val, m_val, h_val], axis=-1)

        t_train = np.argmax(t_train, axis=-1)
        t_val = np.argmax(t_val, axis=-1)

        self.model.fit(
            x=[concatenated_covariates, t_train],
            y=y_train,
            epochs=self.num_epochs,
            validation_data=([concatenated_covariates_val, t_val], y_val),
            callbacks=self.make_callbacks(),
            verbose=2
        )

        info("Resetting to best encountered model at", self.best_model_path, ".")

        # Reset to the best model observed in training.
        weights = ModelFactory.load_weights(os.path.join(self.best_model_path, BaseNeuralNetwork.get_save_file_name()))
        self.model.set_weights(weights)

    @abstractmethod
    def _build(self):
        raise NotImplementedError()

    def make_callbacks(self):
        monitor_name = "val_loss"
        monitor_mode = "min"

        info("Using early stopping on the main loss.")

        callbacks = [
            EarlyStopping(patience=self.early_stopping_patience,
                          monitor=monitor_name,
                          mode=monitor_mode,
                          min_delta=0.0001),
            ModelFactoryCheckpoint(filepath=os.path.join(self.best_model_path, BaseNeuralNetwork.get_save_file_name()),
                                   save_best_only=True,
                                   save_weights_only=True,
                                   monitor=monitor_name,
                                   mode=monitor_mode),
        ]
        return callbacks

    def get_config(self):
        config = {
            "input_shape": self.input_shape,
            "output_dim": self.output_dim,
            "l2_weight": self.l2_weight,
            "num_units": self.num_units,
            "learning_rate": self.learning_rate,
            "p_dropout": self.p_dropout,
            "num_layers": self.num_layers,
            "with_bn": self.with_bn,
            "num_epochs": self.num_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "verbose": self.verbose,
            "batch_size": self.batch_size,
            "best_model_path": self.best_model_path,
            "num_treatments": self.num_treatments,
        }
        return config

    @staticmethod
    def get_config_file_name():
        return "BaseNeuralNetwork_config.json"

    @staticmethod
    def get_subclass_kwargs(config):
        return {}

    @staticmethod
    def load(save_folder_path, base_class=None):
        if base_class is None:
            base_class = BaseNeuralNetwork

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
        subclass_kwargs = base_class.get_subclass_kwargs(config)

        instance = base_class(
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
            **subclass_kwargs
        )
        weight_list = ModelFactory.load_weights(os.path.join(best_model_path, BaseNeuralNetwork.get_save_file_name()))
        instance.model = instance._build()
        instance.model.set_weights(weight_list)
        return instance

    def save(self, save_folder_path, overwrite=True):
        BaseModel.save_config(save_folder_path, self.get_config(), BaseNeuralNetwork.get_config_file_name(), overwrite,
                              BaseNeuralNetwork)
        ModelFactory.save_weights(self.model, os.path.join(save_folder_path, BaseNeuralNetwork.get_save_file_name()))

    @staticmethod
    def get_save_file_name():
        return "model.npz"
