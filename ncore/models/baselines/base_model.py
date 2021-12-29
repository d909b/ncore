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
import os
import sys
import six
import json
from sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod

if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


@six.add_metaclass(ABCMeta)
class BaseModel(object):
    def __init__(self, num_treatments):
        self.num_treatments = num_treatments

    @abstractmethod
    def predict(self, x, m, h, t, s):
        """
        Estimates the potential outcome __y__ from the covariates __x__ the pre-treatment mutations,
        the treatment history __h__, the treatment indices __t__ and the treatment dosages __s__.

        :param x: The pre-treatment covariates. shape = (batch_size, num_covariates)
        :param m: The pre-treatment mutations. shape = (batch_size, num_mutations)
        :param h: The treatment history. Indicators: 0 if has been applied, 1 if treatment __t_i__ at index __i__ has
                  not been applied previously. shape = (batch_size, num_historical_treatments)
                  NOTE: num_historical_treatments is not necessarily equal to num_treatments.
        :param t: The treatment(s) to be applied. Indicators: 0 if not applied, 1 if treatment __t_i__ at index __i__ was
                  applied. shape = (batch_size, num_treatments)
        :param s: The dosage(s) of __t__ to be applied. shape = (batch_size, num_treatments)
                  NOTE: Unused at this point.
        :return: y The estimated potential outcome __y__ after applying treatments __t__ at dosages __s__.
        """
        raise NotImplementedError()

    @abstractmethod
    def fit(self, x_train, m_train, h_train, t_train, s_train, y_train,
                  x_val, m_val, h_val, t_val, s_val, y_val):
        """
        Fit your model given input variables to estimate the potential outcome __y__.
        You have access to training and validation data that you may or may not use to cross-validate your parameters.

        :param x_train:
        :param m_train:
        :param h_train:
        :param t_train:
        :param s_train:
        :param y_train:
        :param x_val:
        :param m_val:
        :param h_val:
        :param t_val:
        :param s_val:
        :param y_val:
        :return: history A history object summarising the intra-training statistics (e.g. loss).
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def load(file_path):
        raise NotImplementedError()

    @abstractmethod
    def save(self, file_path):
        raise NotImplementedError()

    @staticmethod
    def get_save_file_type():
        raise NotImplementedError()

    @staticmethod
    def save_config(directory_path, config, config_file_name, overwrite, outer_class):
        already_exists_exception_message = "__directory_path__ already contains a saved" + outer_class.__name__ + \
                                           " instance and __overwrite__ was set to __False__. Conflicting file: {}"
        config_file_path = os.path.join(directory_path, config_file_name)
        if os.path.exists(config_file_path) and not overwrite:
            raise ValueError(already_exists_exception_message.format(config_file_path))
        else:
            with open(config_file_path, "w") as fp:
                json.dump(config, fp)


class PickleableBaseModel(BaseModel):
    @staticmethod
    def get_save_file_name():
        return "model.pickle"

    @staticmethod
    def load(save_folder_path):
        with open(os.path.join(save_folder_path, PickleableBaseModel.get_save_file_name()), "rb") as load_file:
            return pickle.load(load_file)

    def save(self, save_folder_path):
        with open(os.path.join(save_folder_path, PickleableBaseModel.get_save_file_name()), "wb") as save_file:
            pickle.dump(self, save_file, pickle.HIGHEST_PROTOCOL)


class HyperparamMixin(BaseEstimator):
    @staticmethod
    def get_hyperparameter_ranges():
        raise NotImplementedError()
