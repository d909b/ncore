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
import json
import torch
import numpy as np
from time import time
import torch.nn as nn
from abc import ABCMeta
import torch.optim as optim
from ncore.apps.util import info
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from ncore.apps.util import convert_indicator_list_to_int
from ncore.models.baselines.base_model import BaseModel, HyperparamMixin
from ncore.models.baselines.base_neural_network import BaseNeuralNetwork
from ncore.models.baselines.core_base.relation_tarnet import RelationTARNET
from ncore.models.baselines.core_base.balanced_batch_sampler import BalancedBatchSampler


@six.add_metaclass(ABCMeta)
class CounterfactualRelationEstimator(BaseModel, HyperparamMixin):
    def __init__(self, num_treatments=0, early_stopping_patience=13, best_model_path="", batch_size=32, num_epochs=100,
                 input_shape=(1,), output_dim=1, p_dropout=0.0, l2_weight=0.0, learning_rate=0.001, num_units=128,
                 num_layers=2, with_bn=False, verbose=2, do_split_mixed=True):
        super(CounterfactualRelationEstimator, self).__init__(num_treatments)
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
        self.do_split_mixed = do_split_mixed
        self.best_model_path = best_model_path
        self.early_stopping_patience = early_stopping_patience

    @staticmethod
    def get_hyperparameter_ranges():
        ranges = BaseNeuralNetwork.get_hyperparameter_ranges()
        return ranges

    def predict(self, x, m, h, t, s):
        if self.model is None:
            self.model = self._build()

        concatenated_covariates = np.concatenate([x, m, h], axis=-1)
        y_pred = self.model([
            torch.from_numpy(concatenated_covariates.astype(np.float32)),
            torch.from_numpy(t.astype(np.float32))
        ]).detach().numpy()
        return y_pred

    def _make_loader(self, concatenated_covariates, balancing_scores=None, treatment_assignments=None,
                     ball_trees=None, backlink_indices=None):
        if ball_trees is not None:
            batch_sampler = BalancedBatchSampler(
                RandomSampler(concatenated_covariates),
                self.batch_size,
                drop_last=False,
                ball_trees=ball_trees,
                original_data=concatenated_covariates,
                balancing_scores=balancing_scores,
                backlink_indices=backlink_indices,
                treatment_assignments=treatment_assignments
            )
            shuffle = False
            batch_size = 1
        else:
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
        self.fit(
            x_train, m_train, h_train, t_train, s_train, y_train,
            x_val, m_val, h_val, t_val, s_val, y_val, ball_trees=None, balancing_scores=None, backlink_indices=None
        )

    def fit(self, x_train, m_train, h_train, t_train, s_train, y_train,
            x_val, m_val, h_val, t_val, s_val, y_val, ball_trees=None, balancing_scores=None, backlink_indices=None):
        if self.model is None:
            self.model = self._build()

        model_file_path = os.path.join(self.best_model_path, CounterfactualRelationEstimator.get_save_file_name())

        # Save once up front in case training does not converge.
        torch.save(self.model.state_dict(), model_file_path)

        concatenated_covariates = np.concatenate(
            [x_train, m_train, h_train, t_train, y_train.reshape((-1, 1))],
            axis=-1
        ).astype(np.float32)
        concatenated_covariates_val = np.concatenate(
            [x_val, m_val, h_val, t_val, y_val.reshape((-1, 1))],
            axis=-1
        ).astype(np.float32)

        if self.do_split_mixed:
            train_pure_indices = np.where(t_train.sum(axis=-1) == 1)[0]
            train_mixed_indices = np.where(t_train.sum(axis=-1) != 1)[0]
            train_loaders = [
                self._make_loader(concatenated_covariates[train_pure_indices]),
                self._make_loader(concatenated_covariates[train_mixed_indices])
            ]
            train_prepare_funs = [lambda: self.model.train_pure(), lambda: self.model.train_mixed()]
        else:
            loader_train = self._make_loader(
                concatenated_covariates,
                treatment_assignments=t_train,
                balancing_scores=balancing_scores,
                ball_trees=ball_trees,
                backlink_indices=backlink_indices
            )
            train_loaders = [loader_train]
            train_prepare_funs = [lambda: 0]

        loader_val = self._make_loader(concatenated_covariates_val)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_val_loss, num_epochs_no_improvement = float("inf"), 0
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            self.model.train()
            start_time = time()

            train_loss, num_batches_seen = 0.0, 0
            for loader, prepare_fun in zip(train_loaders, train_prepare_funs):
                prepare_fun()

                for i, batch_data in enumerate(loader):
                    inputs, treatments, labels = \
                        batch_data[:, :-1-t_train.shape[-1]], \
                        batch_data[:, -1-t_train.shape[-1]:-1].reshape((-1, t_train.shape[-1])),\
                        batch_data[:, -1]

                    optimizer.zero_grad()

                    outputs = self.model([inputs, treatments])
                    loss = torch.mean(torch.square(outputs - labels))

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    num_batches_seen += 1
            train_loss /= num_batches_seen

            self.model.eval()
            val_loss, num_batches_seen_val = 0.0, 0
            for batch_data in loader_val:
                inputs, treatments, labels = \
                    batch_data[:, :-1-t_train.shape[-1]], \
                    batch_data[:, -1-t_train.shape[-1]:-1].reshape((-1, t_train.shape[-1])),\
                    batch_data[:, -1]

                outputs = self.model([inputs, treatments])
                loss = criterion(outputs, labels)
                val_loss += loss
                num_batches_seen_val += 1
            val_loss /= num_batches_seen_val

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), model_file_path)
                num_epochs_no_improvement = 0
            else:
                num_epochs_no_improvement += 1

            epoch_duration = time() - start_time
            info("Epoch {:d}/{:d} [{:.2f}s]: loss = {:.4f}, val_loss = {:.4f}"
                 .format(epoch, self.num_epochs, epoch_duration, train_loss, val_loss))

            if num_epochs_no_improvement >= self.early_stopping_patience:
                break

        info("Resetting to best encountered model at", model_file_path, ".")

        # Reset to the best model observed in training.
        self.model.load_state_dict(torch.load(model_file_path))

    def _build(self):
        return RelationTARNET(
            self.input_shape[-1], self.num_treatments, self.num_layers, self.num_units, self.num_layers, self.num_units
        )

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
        return "CounterfactualRelationEstimator_config.json"

    @staticmethod
    def load(save_folder_path):
        config_file_name = CounterfactualRelationEstimator.get_config_file_name()
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

        instance = CounterfactualRelationEstimator(
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
            num_treatments=num_treatments
        )
        weight_list = torch.load(os.path.join(best_model_path, CounterfactualRelationEstimator.get_save_file_name()))
        instance.model = instance._build()
        instance.model.load_state_dict(weight_list)
        return instance

    def save(self, save_folder_path, overwrite=True):
        BaseModel.save_config(save_folder_path, self.get_config(), self.get_config_file_name(), overwrite,
                              CounterfactualRelationEstimator)
        torch.save(
            self.model.state_dict(),
            os.path.join(save_folder_path, CounterfactualRelationEstimator.get_save_file_name())
        )

    @staticmethod
    def get_save_file_name():
        return "model.pt"


class CounterfactualRelationEstimatorNoMixing(CounterfactualRelationEstimator):
    def __init__(self, num_treatments=0, early_stopping_patience=13, best_model_path="", batch_size=32, num_epochs=100,
                 input_shape=(1,), output_dim=1, p_dropout=0.0, l2_weight=0.0, learning_rate=0.001, num_units=128,
                 num_layers=2, with_bn=False, verbose=2):
        super(CounterfactualRelationEstimatorNoMixing, self).__init__(
            num_treatments=num_treatments, early_stopping_patience=early_stopping_patience, best_model_path=best_model_path,
            batch_size=batch_size, num_epochs=num_epochs, input_shape=input_shape, output_dim=output_dim,
            p_dropout=p_dropout, l2_weight=l2_weight, learning_rate=learning_rate, num_units=num_units,
            num_layers=num_layers, with_bn=with_bn, verbose=verbose, do_split_mixed=False
        )


class BalancedCounterfactualRelationEstimator(CounterfactualRelationEstimatorNoMixing):
    def __init__(self, num_treatments=0, early_stopping_patience=13, best_model_path="", batch_size=32, num_epochs=100,
                 input_shape=(1,), output_dim=1, p_dropout=0.0, l2_weight=0.0, learning_rate=0.001, num_units=128,
                 num_layers=2, with_bn=False, verbose=2, balancing_score_dim=8):
        super(BalancedCounterfactualRelationEstimator, self).__init__(
            num_treatments=num_treatments, early_stopping_patience=early_stopping_patience, best_model_path=best_model_path,
            batch_size=batch_size, num_epochs=num_epochs, input_shape=input_shape, output_dim=output_dim,
            p_dropout=p_dropout, l2_weight=l2_weight, learning_rate=learning_rate, num_units=num_units,
            num_layers=num_layers, with_bn=with_bn, verbose=verbose
        )
        self.balancing_score_dim = balancing_score_dim

    def fit(self, x_train, m_train, h_train, t_train, s_train, y_train,
            x_val, m_val, h_val, t_val, s_val, y_val):
        concatenated_covariates = np.concatenate(
            [x_train, m_train, h_train, t_train, y_train.reshape((-1, 1))],
            axis=-1
        ).astype(np.float32)
        concatenated_covariates_val = np.concatenate(
            [x_val, m_val, h_val, t_val, y_val.reshape((-1, 1))],
            axis=-1
        ).astype(np.float32)

        clf = PCA(n_components=self.balancing_score_dim, random_state=0, svd_solver="randomized", whiten=True)

        balancing_scores = clf.fit_transform(concatenated_covariates)
        t_train_converted = np.array([convert_indicator_list_to_int(t) for t in t_train])

        all_indices = np.arange(len(balancing_scores))
        ball_trees, backlink_indices = {}, {}
        for t in set(t_train_converted):
            ball_trees[t] = BallTree(balancing_scores[t_train_converted == t])
            backlink_indices[t] = all_indices[t_train_converted == t]
        super(BalancedCounterfactualRelationEstimator, self).fit(
            x_train, m_train, h_train, t_train, s_train, y_train,
            x_val, m_val, h_val, t_val, s_val, y_val,
            ball_trees=ball_trees, balancing_scores=balancing_scores, backlink_indices=backlink_indices
        )
