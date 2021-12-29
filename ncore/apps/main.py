#!/usr/bin/env python
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import glob
import inspect
import importlib
import numpy as np
import pandas as pd
from os.path import join
from sklearn.pipeline import Pipeline
from ncore.apps.util import info, warn, error
from ncore.apps.parameters import parse_parameters
from ncore.apps.base_application import BaseApplication
from ncore.models.model_evaluation import ModelEvaluation
from ncore.apps.util import time_function, clip_percentage
from ncore.data_access.simulator_data_loader import SimulatorDataLoader
from ncore.data_access.generator import make_generator, get_last_row_id
from ncore.data_access.crispr3way_data_loader import CRISPR3WayDataLoader
from ncore.data_access.europe_1_data_loader import Europe1DataLoader, SemisyntheticEurope1DataLoader
from ncore.data_access.europe_2_data_loader import Europe2DataLoader, SemisyntheticEurope2DataLoader


if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


class MainApplication(BaseApplication):
    def __init__(self, args):
        super(MainApplication, self).__init__(args)
        self.training_set, self.validation_set, self.test_set, self.input_shape, self.num_treatments, \
          self.output_dim, self.feature_names, self.feature_types, self.output_names, self.output_types, \
            self.x_preprocessors, self.m_preprocessors, self.output_preprocessors = [None]*13
        self.loader = None
        self.visualisations_output_directory = None
        self.load_data()

    def load_data(self):
        seed = int(np.rint(self.args["seed"]))
        resample_with_replacement = self.args["resample_with_replacement"]

        self.training_set, self.validation_set, self.test_set, self.input_shape, self.output_dim, \
          self.feature_names, self.feature_types, self.output_names, self.output_types, self.num_treatments, \
            self.x_preprocessors, self.m_preprocessors, self.output_preprocessors = \
              self.get_data(seed=seed, resample=resample_with_replacement, resample_seed=seed)

    def setup(self):
        super(MainApplication, self).setup()

    def get_data(self, seed=0, resample=False, resample_seed=0):
        dataset = self.args["dataset"].lower()
        output_directory = self.args["output_directory"]
        num_simulated_patients = int(np.rint(self.args["num_simulated_patients"]))
        num_simulated_treatments = int(np.rint(self.args["num_simulated_treatments"]))
        treatment_assignment_bias_coefficient = float(self.args["treatment_assignment_bias_coefficient"])

        self.visualisations_output_directory = os.path.join(output_directory, "visualisations")
        if not os.path.exists(self.visualisations_output_directory):
            os.mkdir(self.visualisations_output_directory)

        x_steps, m_steps, output_steps = None, None, None

        if dataset.lower().startswith("crispr3way"):
            achilles_dataset_path = \
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "../../crisprko-3way/achilles-gene-effect_20q4.h5")
            crispr3way_dataset_path = \
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "./../../crisprko-3way/crispr-3-way.csv")
            crispr3way_gene_list_path = \
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "./../../crisprko-3way/gene-list-crispr-3-way.csv")
            loader = CRISPR3WayDataLoader(achilles_dataset_path, crispr3way_dataset_path, crispr3way_gene_list_path)
        elif dataset.lower().startswith("europe1"):
            europe1_dataset_path = \
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "./../../europe1/tce-dose-all.csv")

            if dataset.lower() == "europe1":
                loader = Europe1DataLoader(europe1_dataset_path)
            else:
                loader = SemisyntheticEurope1DataLoader(europe1_dataset_path,
                                                        output_directory=self.visualisations_output_directory)
        elif dataset.lower().startswith("europe2"):
            europe2_dataset_path = \
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "./../../europe2/Europe2.RData")
            if dataset.lower() == "europe2":
                loader = Europe2DataLoader(europe2_dataset_path)
            else:
                loader = SemisyntheticEurope2DataLoader(europe2_dataset_path,
                                                        output_directory=self.visualisations_output_directory)
        elif dataset.lower() == "simulator":
            loader = SimulatorDataLoader(seed, output_directory=self.visualisations_output_directory,
                                         num_simulated_treatments=num_simulated_treatments,
                                         num_simulated_patients=num_simulated_patients,
                                         treatment_assignment_bias_coefficient=treatment_assignment_bias_coefficient)
        else:
            raise NotImplementedError("{:s} is not a valid dataset.".format(dataset))
        self.loader = loader
        return self.loader.get_data(self.args, seed=seed, do_resample=resample, resample_seed=resample_seed,
                                    x_steps=x_steps, m_steps=m_steps, output_steps=output_steps)

    def get_num_losses(self):
        return 1

    def make_train_generator(self, randomise=True, stratify=True):
        batch_size = int(np.rint(self.args["batch_size"]))
        seed = int(np.rint(self.args["seed"]))
        num_losses = self.get_num_losses()

        train_generator, train_steps = make_generator(dataset=self.training_set,
                                                      batch_size=batch_size,
                                                      num_losses=num_losses,
                                                      shuffle=randomise,
                                                      seed=seed)

        return train_generator, train_steps

    def make_validation_generator(self, randomise=False):
        batch_size = int(np.rint(self.args["batch_size"]))
        num_losses = self.get_num_losses()

        val_generator, val_steps = make_generator(dataset=self.validation_set,
                                                  batch_size=batch_size,
                                                  num_losses=num_losses,
                                                  shuffle=randomise)
        return val_generator, val_steps

    def make_test_generator(self, randomise=False, do_not_sample_equalised=False):
        batch_size = int(np.rint(self.args["batch_size"]))
        num_losses = self.get_num_losses()

        test_generator, test_steps = make_generator(dataset=self.test_set,
                                                    batch_size=batch_size,
                                                    num_losses=num_losses,
                                                    shuffle=randomise)
        return test_generator, test_steps

    def get_best_model_path(self):
        return join(self.args["output_directory"], "model")

    def get_preprocessor_path_x(self):
        return join(self.args["output_directory"], "preprocessor_x.pickle")

    def get_preprocessor_path_m(self):
        return join(self.args["output_directory"], "preprocessor_m.pickle")

    def get_output_preprocessor_path(self):
        return join(self.args["output_directory"], "output_preprocessor.pickle")

    def get_prediction_path(self, set_name):
        return join(self.args["output_directory"], set_name + "_predictions.tsv")

    def get_thresholded_prediction_path(self, set_name):
        return join(self.args["output_directory"], set_name + "_predictions.thresholded.tsv")

    def get_hyperopt_parameters(self):
        hyper_params = {}

        resample_with_replacement = self.args["resample_with_replacement"]
        if resample_with_replacement:
            base_params = {
                "seed": [0, 2**32-1],
            }
        else:
            cls = self.get_model_type_for_method_name()
            if cls is not None:
                base_params = cls.get_hyperparameter_ranges()
            else:
                warn("Unable to retrieve class for provided method name [", self.args["method"], "].")
                base_params = {}

        hyper_params.update(base_params)
        return hyper_params

    def get_model_type_for_method_name(self):
        from ncore.models.baselines.base_model import BaseModel

        method = self.args["method"]
        baseline_package_path = os.path.dirname(inspect.getfile(BaseModel))

        for module_path in glob.glob(baseline_package_path + "/*.py"):
            modname = os.path.basename(module_path)[:-3]
            fully_qualified_name = BaseModel.__module__
            fully_qualified_name = fully_qualified_name[:fully_qualified_name.rfind(".")] + "." + modname
            mod = importlib.import_module(fully_qualified_name)
            if hasattr(mod, method):
                cls = getattr(mod, method)
                return cls
        return None

    def get_model_for_method_name(self, model_params):
        cls = self.get_model_type_for_method_name()
        if cls is not None:
            instance = cls()
            available_model_params = {k: model_params[k] if k in model_params else instance.get_params()[k]
                                      for k in instance.get_params().keys()}
            instance = instance.set_params(**available_model_params)

            return instance
        else:
            return None

    @time_function("train_model")
    def train_model(self, train_generator, train_steps, val_generator, val_steps):
        info("Started training model.")

        with_bn = self.args["with_bn"]
        with_tensorboard = self.args["with_tensorboard"]
        output_directory = self.args["output_directory"]
        n_jobs = int(np.rint(self.args["n_jobs"]))
        num_epochs = int(np.rint(self.args["num_epochs"]))
        learning_rate = float(self.args["learning_rate"])
        l2_weight = float(self.args["l2_weight"])
        batch_size = int(np.rint(self.args["batch_size"]))
        early_stopping_patience = int(np.rint(self.args["early_stopping_patience"]))
        num_layers = int(np.rint(self.args["num_layers"]))
        num_units = int(np.rint(self.args["num_units"]))
        svm_c = float(self.args["svm_c"])
        dropout = float(self.args["dropout"])
        n_estimators = int(np.rint(self.args["n_estimators"]))
        max_depth = int(np.rint(self.args["max_depth"]))
        use_multi_gpu = self.args["use_multi_gpu"]
        best_model_path = self.get_best_model_path()
        seed = int(np.rint(self.args["seed"]))

        os.makedirs(best_model_path, exist_ok=True)

        model_params = {
            "output_directory": output_directory,
            "early_stopping_patience": early_stopping_patience,
            "num_layers": num_layers,
            "num_units": num_units,
            "p_dropout": dropout,
            "input_shape": self.input_shape,
            "output_dim": self.output_dim,
            "output_types": self.output_types,
            "output_preprocessors": self.output_preprocessors,
            "num_treatments": self.num_treatments,
            "visualisations_output_directory": self.visualisations_output_directory,
            "batch_size": batch_size,
            "best_model_path": best_model_path,
            "l2_weight": l2_weight,
            "learning_rate": learning_rate,
            "with_bn": with_bn,
            "with_tensorboard": with_tensorboard,
            "n_jobs": n_jobs,
            "num_epochs": num_epochs,
            "C": svm_c,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "multi_device": use_multi_gpu,
            "seed": seed,
            "random_state": seed,
        }

        assert train_steps > 0, "You specified a batch_size that is bigger than the size of the train set."
        assert val_steps > 0, "You specified a batch_size that is bigger than the size of the validation set."

        if with_tensorboard:
            tb_folder = join(self.args["output_directory"], "tensorboard")
            tmp_generator, tmp_steps = val_generator, val_steps
            tb = [MainApplication.build_tensorboard(tmp_generator, tb_folder)]
        else:
            tb = []

        model_params["tensorboard_callback"] = tb

        if self.args["load_existing"]:
            info("Loading existing model from", self.args["load_existing"])
            model_class = self.get_model_type_for_method_name()
            model = model_class.load(self.args["load_existing"])
        else:
            model = self.get_model_for_method_name(model_params)

        if self.args["do_train"]:
            if hasattr(model, "fit_generator"):
                history = model.fit_generator(train_generator, train_steps, val_generator, val_steps,
                                              self.input_shape, self.output_dim)
            else:
                (x_train, y_train, t_train, p_train, h_train, m_train, s_train), \
                 (x_val, y_val, t_val, p_val, h_val, m_val, s_val) = self.training_set, self.validation_set

                history = model.fit(x_train, m_train, h_train, t_train, s_train, y_train,
                                    x_val, m_val, h_val, t_val, s_val, y_val)

            if isinstance(model, Pipeline):
                model.steps[1][1].save(best_model_path)
            else:
                model.save(best_model_path)

            info("Saved model to", best_model_path)

            x_preprocessor_path = self.get_preprocessor_path_x()
            m_preprocessor_path = self.get_preprocessor_path_m()
            output_preprocessor_path = self.get_output_preprocessor_path()

            with open(x_preprocessor_path, "wb") as fp:
                pickle.dump(self.x_preprocessors, fp)
            info("Saved demographics preprocessors to", x_preprocessor_path)

            with open(m_preprocessor_path, "wb") as fp:
                pickle.dump(self.m_preprocessors, fp)
            info("Saved mutation preprocessors to", m_preprocessor_path)

            with open(output_preprocessor_path, "wb") as fp:
                pickle.dump(self.output_preprocessors, fp)
            info("Saved output preprocessors to", output_preprocessor_path)
        else:
            history = {
                "val_acc": [],
                "val_loss": [],
                "val_combined_loss": [],
                "acc": [],
                "loss": [],
                "combined_loss": []
            }
        return model, history

    @time_function("evaluate_model")
    def evaluate_model(self, model, test_generator, test_steps, with_print=True, set_name="test", threshold=None):
        dataset = self.args["dataset"]
        output_directory = self.args["output_directory"]

        if with_print:
            info("Started evaluation.")

        scores = ModelEvaluation.evaluate(model, test_generator, test_steps, set_name,
                                          threshold=threshold, with_print=with_print, loader=self.loader,
                                          output_preprocessors=self.output_preprocessors,
                                          output_directory=output_directory,
                                          dataset=dataset)
        return scores

    def save_predictions(self, model, threshold=None):
        info("Saving model predictions.")

        fraction_of_data_set = clip_percentage(self.args["fraction_of_data_set"])

        generators = [self.make_train_generator, self.make_validation_generator, self.make_test_generator]
        generator_names = ["train", "val", "test"]
        for generator_fun, generator_name in zip(generators, generator_names):
            generator, steps = generator_fun(randomise=False)
            steps = int(np.rint(steps * fraction_of_data_set))

            predictions = []
            for step in range(steps):
                x, m, h, t, s, y = next(generator)
                last_id = get_last_row_id()
                y_pred = model.predict(x, m, h, t, s)
                if len(y.shape) == 1 and len(y_pred.shape) == 2:
                    y_pred = y_pred[:, 0]
                y_pred = np.squeeze(y_pred)
                if y_pred.size == 1:
                    y_pred = [y_pred]

                for current_id, current_y_pred in zip(last_id, y_pred):
                    predictions.append([current_id, current_y_pred])
            row_ids = np.hstack(list(map(lambda x: x[0], predictions)))
            outputs = np.stack(list(map(lambda x: x[1], predictions)))
            file_path = self.get_prediction_path(generator_name)

            num_predictions = 1 if len(outputs.shape) == 1 else outputs.shape[-1]
            assert num_predictions == len(self.output_names)

            columns = self.output_names

            df = pd.DataFrame(outputs, columns=columns, index=row_ids)
            df.index.name = "PATIENTID"
            df.to_csv(file_path, sep="\t")
            info("Saved raw model predictions to", file_path)

            if threshold is not None:
                thresholded_file_path = self.get_thresholded_prediction_path(generator_name)
                df = pd.DataFrame((outputs > threshold).astype(int), columns=columns, index=row_ids)
                df.index.name = "PATIENTID"
                df.to_csv(thresholded_file_path, sep="\t")
                info("Saved thresholded model predictions to", thresholded_file_path)

    @staticmethod
    def build_tensorboard(tmp_generator, tb_folder):
        from tensorflow.keras.callbacks import TensorBoard

        for a_file in os.listdir(tb_folder):
            file_path = join(tb_folder, a_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                error(e)

        tb = TensorBoard(tb_folder,
                         write_graph=False,
                         histogram_freq=0,
                         write_grads=False,
                         write_images=False,
                         update_freq='epoch')
        # tb.validation_data[1] = np.expand_dims(tb.validation_data[1], axis=-1)
        # if isinstance(y, list):
        #     num_targets = len(y)
        #     tb.validation_data += [y[0]] + y[1:]
        # else:
        #     tb.validation_data += [y]
        #     num_targets = 1
        #
        # tb.validation_data += [np.ones(x[0].shape[0])] * num_targets + [0.0]
        return tb


if __name__ == '__main__':
    app = MainApplication(parse_parameters())
    app.run()
