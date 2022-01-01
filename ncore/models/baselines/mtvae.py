"""
Copyright (C) 2021  Patrick Schwab, GlaxoSmithKline plc, Sonali Parbhoo, Harvard University

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

MIT License (SOURCE: https://github.com/jma712/DIRECT)

Copyright (c) 2021 Jing Ma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import json
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from ncore.models.baselines.mtvae_base.mtvae import MTVAE as MTVAEBase
from ncore.models.baselines.base_model import BaseModel, HyperparamMixin
from ncore.models.baselines.base_neural_network import BaseNeuralNetwork


class MTVAE(BaseModel, HyperparamMixin):
    def __init__(self, num_treatments=0, best_model_path="", batch_size=32, num_epochs=150,
                 input_shape=(1,), l2_weight=0.0, learning_rate=0.001, num_units=128, verbose=2):
        super(MTVAE, self).__init__(num_treatments)
        self.model = None
        self.verbose = verbose
        self.input_treat = None
        self.l2_weight = l2_weight
        self.num_units = num_units
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.best_model_path = best_model_path

    @staticmethod
    def get_hyperparameter_ranges():
        ranges = BaseNeuralNetwork.get_hyperparameter_ranges()
        return ranges

    def predict(self, x, m, h, t, s):
        if self.model is None:
            raise AssertionError("Model must be __fit__ before calling __predict__.")

        y_pred = self.model.predictY(
            torch.from_numpy(t.astype(np.float32)),
            torch.from_numpy(self.input_treat.T.astype(np.float32)),
        )[0].detach().numpy()
        return y_pred

    def _make_loader(self, concatenated_covariates):
        batch_sampler = None
        shuffle = True
        batch_size = self.batch_size
        loader = DataLoader(
            concatenated_covariates,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=None,
            batch_sampler=batch_sampler,
            num_workers=0,
            collate_fn=None,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
        )
        return loader

    def fit(self, x_train, m_train, h_train, t_train, s_train, y_train,
            x_val, m_val, h_val, t_val, s_val, y_val):
        if self.model is None:
            self.model = self._build(t_train)

        model_file_path = os.path.join(self.best_model_path, MTVAE.get_save_file_name())

        # Save once up front in case training does not converge.
        torch.save(self.model.state_dict(), model_file_path)

        concatenated_covariates = np.concatenate(
            [x_train, m_train, h_train, t_train, y_train.reshape((-1, 1))],
            axis=-1
        ).astype(np.float32)
        loader_train = self._make_loader(
            concatenated_covariates
        )
        input_treat_trn = torch.Tensor(t_train.T.astype(np.float32))

        par_t = list(self.model.mu_zt.parameters()) + list(self.model.logvar_zt.parameters()) + list(
            self.model.a_reconstby_zt.parameters())
        par_z = list(self.model.mu_zi_k.parameters()) + list(self.model.logvar_zi_k.parameters())
        par_y = list(self.model.y_pred_1.parameters()) + list(self.model.y_pred_2.parameters()) + par_z
        optimizer_1 = optim.Adam([{'params': par_t, 'lr': self.learning_rate}], weight_decay=self.l2_weight)  # zt
        optimizer_2 = optim.Adam([{'params': [self.model.mu_p_zt, self.model.logvar_p_zt], 'lr': 0.01}],
                                 weight_decay=self.l2_weight)  # centroid
        optimizer_3 = optim.Adam([{'params': par_z, 'lr': self.learning_rate}], weight_decay=self.l2_weight)  # zi
        optimizer_4 = optim.Adam([{'params': par_y, 'lr': self.learning_rate}], weight_decay=self.l2_weight)  # y
        optimizer = [optimizer_1, optimizer_2, optimizer_3, optimizer_4]
        self.model.fit(self.num_epochs, self.num_treatments, loader_train, input_treat_trn, optimizer)

    def _build(self, input_treat):
        self.input_treat = input_treat
        return MTVAEBase(
            num_treatments=self.num_treatments,
            num_train_samples=len(input_treat),
            dim_zi=self.num_units,
            dim_zt=self.num_units
        )

    def get_config(self):
        config = {
            "input_shape": self.input_shape,
            "num_units": self.num_units,
            "l2_weight": self.l2_weight,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "verbose": self.verbose,
            "batch_size": self.batch_size,
            "best_model_path": self.best_model_path,
            "num_treatments": self.num_treatments,
        }
        return config

    @staticmethod
    def get_config_file_name():
        return "MTVAE_config.json"

    @staticmethod
    def load(save_folder_path):
        config_file_name = MTVAE.get_config_file_name()
        config_file_path = os.path.join(save_folder_path, config_file_name)
        with open(config_file_path, "r") as fp:
            config = json.load(fp)

        input_shape = config["input_shape"]
        l2_weight = config["l2_weight"]
        num_units = config["num_units"]
        learning_rate = config["learning_rate"]
        num_epochs = config["num_epochs"]
        verbose = config["verbose"]
        batch_size = config["batch_size"]
        num_treatments = config["num_treatments"]
        best_model_path = config["best_model_path"]

        instance = MTVAE(
            input_shape=input_shape,
            l2_weight=l2_weight,
            num_units=num_units,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            verbose=verbose,
            batch_size=batch_size,
            best_model_path=best_model_path,
            num_treatments=num_treatments
        )
        instance.input_treat = np.load(os.path.join(save_folder_path, MTVAE.get_array_save_file_name()))
        weight_list = torch.load(os.path.join(best_model_path, MTVAE.get_save_file_name()))
        instance.model = instance._build(instance.input_treat)
        instance.model.load_state_dict(weight_list)
        return instance

    def save(self, save_folder_path, overwrite=True):
        BaseModel.save_config(save_folder_path, self.get_config(), self.get_config_file_name(), overwrite,
                              MTVAE)
        np.save(os.path.join(save_folder_path, MTVAE.get_array_save_file_name()), self.input_treat)
        torch.save(
            self.model.state_dict(),
            os.path.join(save_folder_path, MTVAE.get_save_file_name())
        )

    @staticmethod
    def get_save_file_name():
        return "model.pt"

    @staticmethod
    def get_array_save_file_name():
        return "model.npy"
