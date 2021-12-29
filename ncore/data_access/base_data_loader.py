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
import six
import datetime
import numpy as np
from collections import Counter
from ncore.apps.util import info
from abc import ABCMeta, abstractmethod
from ncore.apps.util import time_function
from ncore.apps.util import clip_percentage
from tensorflow.keras.utils import to_categorical
from ncore.data_access.meta_data.feature_types import FeatureTypeUnknown


@six.add_metaclass(ABCMeta)
class BaseDataLoader(object):
    @abstractmethod
    def get_patients(self):
        raise NotImplementedError()

    @abstractmethod
    def transform_patients(self, patients):
        raise NotImplementedError()

    def generate_outcomes(self, y, p, x, m, h, s, t, x_feature_types,
                          x_train=None, m_train=None, h_train=None, s_train=None, t_train=None):
        return y, t

    @abstractmethod
    def get_covariate_preprocessors(self, seed):
        raise NotImplementedError()

    @abstractmethod
    def get_output_preprocessors(self, seed):
        raise NotImplementedError()

    def preprocess_covariates(self, x, feature_names=None, steps=None, seed=909):
        return BaseDataLoader.preprocess_variables(x, self.get_covariate_preprocessors,
                                                   feature_names=feature_names, steps=steps, seed=seed)

    def preprocess_outputs(self, x, feature_names=None, steps=None, seed=909):
        return BaseDataLoader.preprocess_variables(x, self.get_output_preprocessors,
                                                   feature_names=feature_names, steps=steps, seed=seed)

    @staticmethod
    def preprocess_variables(x, get_fun=None, feature_names=None, steps=None, seed=909, transform_fun_name="transform"):
        load_new = steps is None
        if load_new:
            steps = get_fun(seed=seed)

        prior_feature_types = [FeatureTypeUnknown() for _ in range(x.shape[-1])]
        prior_feature_names = feature_names
        for step in steps:
            step.feature_names = prior_feature_names
            step.feature_types = prior_feature_types
            if load_new:
                step.fit(x)
            x = step.transform(x, transform_fun_name=transform_fun_name)
            prior_feature_names = step.feature_names
            prior_feature_types = step.feature_types

        if feature_names is not None:
            return x, steps, prior_feature_names, prior_feature_types
        else:
            return x, steps

    @abstractmethod
    def get_id_name(self):
        raise NotImplementedError()

    @abstractmethod
    def get_split_property_names(self):
        raise NotImplementedError()

    @abstractmethod
    def get_discrete_covariate_names_and_indices(self):
        raise NotImplementedError()

    @abstractmethod
    def get_continuous_covariate_names_and_indices(self):
        raise NotImplementedError()

    @staticmethod
    def to_dates(x):
        return list(map(lambda xi: None if isinstance(xi[0], float) and np.isnan(xi[0]) else
                                   datetime.datetime.strptime(xi[0], "%d.%m.%y").date(), x))

    @staticmethod
    def counter_to_string(counter):
        count_sum = sum(counter.values())
        output_string = ""
        for i, k in enumerate(sorted(counter.keys())):
            v = counter[k]
            output_string += "{KEY} = {VALUE:.2f}%".format(KEY=str(k).lower(), VALUE=100*float(v)/count_sum)
            if i != len(counter) - 1:
                output_string += ", "
        return output_string

    def report_data_fold(self, x, t, y, set_name):
        get_uncertainty = lambda x: str(np.median(x)) + \
                                    " (" + str(np.quantile(x, 0.1)) + \
                                    ", " + str(np.quantile(x, 0.9)) + ")"

        covariate_names, covariate_indices = self.get_discrete_covariate_names_and_indices()
        continuous_names, continuous_indices = self.get_continuous_covariate_names_and_indices()
        counters = []
        for j in covariate_indices:
            if j < 0:
                counter = Counter(t[:, j+1])
            else:
                counter = Counter(x[:, j])
            counters.append(counter)
        info("Using ", set_name, " set with n = ", len(x),
             " with Covariates: {} {}"
             .format(list(map(lambda name, counts: "{} = [{}]".format(name, BaseDataLoader.counter_to_string(counts)),
                              covariate_names, counters)),
                     list(map(lambda name, index: "{} = {}".format(name, get_uncertainty(x[:, index])),
                              continuous_names, continuous_indices))),
             sep="")

    @staticmethod
    def make_synthetic_labels_for_stratification(num_bins=5, min_synthetic_group_size=100, max_num_unique_values=10,
                                                 label_candidates=list([])):
        synthetic_labels = []
        for arg_idx in range(len(label_candidates)):
            arg = label_candidates[arg_idx]
            if len(arg) == 0:
                raise ValueError("Length of synthetic label inputs should not be zero.")

            if len(set(arg)) <= max_num_unique_values:
                more_labels = to_categorical(np.array(arg).astype(int), num_classes=len(set(arg)))
            else:
                assignments = np.digitize(arg, np.linspace(np.min(arg), np.max(arg), num_bins)) - 1

                while True:
                    if len(assignments) < min_synthetic_group_size:
                        break

                    counts = Counter(assignments)
                    has_reset_any = False
                    for key, num_instances in counts.items():
                        if num_instances < min_synthetic_group_size:
                            new_key = key - 1 if key != 0 else key + 1
                            assignments[assignments == key] = new_key
                            has_reset_any = True
                            break
                    if not has_reset_any:
                        break

                more_labels = to_categorical(assignments,
                                             num_classes=num_bins)

            synthetic_labels.append(more_labels)
        synthetic_labels = np.column_stack(synthetic_labels)
        return synthetic_labels

    def split_dataset(self, rows, num_validation_samples, num_test_samples,
                      random_state=909, max_num_unique_values=10):
        split_property_names = self.get_split_property_names()
        if len(split_property_names) == 0:
            synthetic_labels = np.array([[0] for _ in range(len(rows))])
        else:
            split_properties = rows[split_property_names].values
            synthetic_labels = []
            for j in range(split_properties.shape[-1]):
                unique_values = set(split_properties[:, j])
                if len(unique_values) < max_num_unique_values:
                    index_map = dict(zip(unique_values, range(len(unique_values))))
                    labels = list(map(lambda xi: index_map[xi], split_properties[:, j]))
                else:
                    labels = split_properties[:, j]
                synthetic_labels.append(labels)

        x = np.arange(len(rows))

        from skmultilearn.model_selection import iterative_train_test_split

        for bin_size in reversed([2, 3, 4, 5]):
            if len(split_property_names) != 0:
                synthetic_labels = BaseDataLoader.make_synthetic_labels_for_stratification(
                    label_candidates=synthetic_labels,
                    num_bins=bin_size,
                    min_synthetic_group_size=100,
                    max_num_unique_values=max_num_unique_values
                )

            test_fraction = num_test_samples / float(len(synthetic_labels))
            rest_index, _, test_index, _ = iterative_train_test_split(x[:, np.newaxis], synthetic_labels,
                                                                      test_size=test_fraction)
            rest_index, test_index = rest_index[:, 0], test_index[:, 0]

            val_fraction = num_validation_samples / float(len(rest_index))
            train_index, _, val_index, _ = iterative_train_test_split(rest_index[:, np.newaxis],
                                                                      synthetic_labels[rest_index],
                                                                      test_size=val_fraction)
            train_index, val_index = train_index[:, 0], val_index[:, 0]
            break

        assert len(set(train_index).intersection(set(val_index))) == 0
        assert len(set(train_index).intersection(set(test_index))) == 0

        return train_index, val_index, test_index

    @time_function("load_data")
    def get_data(self, args, do_resample=False, seed=0, resample_seed=0,
                 x_steps=None, m_steps=None, output_steps=None):
        patients = self.get_patients()

        num_patients = len(patients)
        patient_ids = np.squeeze(patients[[self.get_id_name()]].values)

        info("Loaded data with", num_patients, "samples.")

        if do_resample:
            random_state = np.random.RandomState(resample_seed)
            resampled_samples = random_state.randint(0, num_patients, size=num_patients)
            patients = patients.iloc[resampled_samples]
            patient_ids = [patient_ids[idx] for idx in resampled_samples]

        test_set_fraction = clip_percentage(args["test_set_fraction"])
        validation_set_fraction = clip_percentage(args["validation_set_fraction"])
        num_test_samples = int(np.rint(test_set_fraction * num_patients))
        num_validation_samples = int(np.rint(validation_set_fraction * num_patients))

        train_index, val_index, test_index = self.split_dataset(patients,
                                                                num_validation_samples,
                                                                num_test_samples)

        train_patients = patients.iloc[train_index]
        p_train, x_train, h_train, m_train, y_train, s_train, t_train, x_names, m_names, output_names = \
            self.transform_patients(train_patients)

        val_patients = patients.iloc[val_index]
        p_val, x_val, h_val, m_val, y_val, s_val, t_val, _, _, _ = self.transform_patients(val_patients)

        test_patients = patients.iloc[test_index]
        p_test, x_test, h_test, m_test, y_test, s_test, t_test, _, _, _ = self.transform_patients(test_patients)

        self.report_data_fold(x_train, t_train, y_train, "training set")
        self.report_data_fold(x_val, t_val, y_val, "validation set")
        self.report_data_fold(x_test, t_test, y_test, "test set")

        if x_steps is None:
            _, x_steps, x_names, x_feature_types = self.preprocess_covariates(x_train, x_names,
                                                                              seed=seed)
        else:
            _, _, x_names, x_feature_types = self.preprocess_covariates(x_train, x_names,
                                                                        steps=x_steps)

        x_train, _ = self.preprocess_covariates(x_train, steps=x_steps)
        x_val, _ = self.preprocess_covariates(x_val, steps=x_steps)
        x_test, _ = self.preprocess_covariates(x_test, steps=x_steps)

        if m_steps is None:
            _, m_steps, m_names, m_feature_types = self.preprocess_covariates(m_train, m_names,
                                                                              seed=seed)
        else:
            _, _, m_names, m_feature_types = self.preprocess_covariates(m_train, m_names,
                                                                        steps=m_steps)

        m_train, _ = self.preprocess_covariates(m_train, steps=m_steps)
        m_val, _ = self.preprocess_covariates(m_val, steps=m_steps)
        m_test, _ = self.preprocess_covariates(m_test, steps=m_steps)

        y_train, t_train = self.generate_outcomes(y_train, p_train, x_train, m_train,
                                                  h_train, s_train, t_train, x_feature_types,
                                                  x_train=x_train, m_train=m_train, h_train=h_train,
                                                  s_train=s_train, t_train=t_train)
        y_val, t_val = self.generate_outcomes(y_val, p_val, x_val, m_val, h_val, s_val, t_val, x_feature_types)
        y_test, t_test = self.generate_outcomes(y_test, p_test, x_test, m_test, h_test, s_test, t_test, x_feature_types)

        if output_steps is None:
            _, output_steps, output_names, output_types = self.preprocess_outputs(y_train, output_names, seed=seed)
        else:
            _, _, output_names, output_types = self.preprocess_outputs(y_train, output_names, steps=output_steps)

        y_train, _ = self.preprocess_outputs(y_train, steps=output_steps)
        y_val, _ = self.preprocess_outputs(y_val, steps=output_steps)
        y_test, _ = self.preprocess_outputs(y_test, steps=output_steps)

        assert x_train.shape[-1] == x_val.shape[-1] and x_val.shape[-1] == x_test.shape[-1]
        assert x_train.shape[-1] == len(x_names)

        input_shape = (x_train.shape[-1] + m_train.shape[-1] + h_train.shape[-1],)

        return (x_train.astype(float), np.squeeze(y_train), np.squeeze(t_train), np.squeeze(p_train), h_train, m_train, s_train), \
               (x_val.astype(float), np.squeeze(y_val), np.squeeze(t_val), np.squeeze(p_val), h_val, m_val, s_val), \
               (x_test.astype(float), np.squeeze(y_test), np.squeeze(t_test), np.squeeze(p_test), h_test, m_test, s_test), \
               input_shape, len(output_names), x_names, x_feature_types, output_names, output_types, t_train.shape[-1], \
               x_steps, m_steps, output_steps
