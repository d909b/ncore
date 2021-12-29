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
import six
import time
import shutil
import numpy as np
from os.path import join
from abc import ABCMeta, abstractmethod
from distutils.dir_util import copy_tree
from ncore.apps.util import info, error, warn


@six.add_metaclass(ABCMeta)
class HyperparameterMixin(object):
    def __init__(self):
        self.best_score_index = 0
        self.best_score = np.finfo(float).min
        self.best_params = ""

    @abstractmethod
    def get_hyperopt_parameters(self):
        return {}

    @staticmethod
    def get_random_hyperopt_parameters(initial_args, hyperopt_parameters, hyperopt_index):
        new_params = dict(initial_args)
        for k, v in hyperopt_parameters.items():
            if isinstance(v, list):
                min_val, max_val = v
                new_params[k] = np.random.uniform(min_val, max_val)
            elif isinstance(v, tuple):
                choice = np.random.choice(v)
                new_params[k] = choice
        return new_params

    @staticmethod
    def calculate_num_hyperparameter_permutations(hyperopt_parameters):
        num_permutations = 1
        for param_range in hyperopt_parameters.values():
            if isinstance(param_range, list):
                return float("inf")
            else:
                num_permutations *= len(param_range)
        return num_permutations

    @staticmethod
    def get_next_hyperopt_choice(initial_args, hyperopt_parameters, state=None):
        if state is None:
            state = [0 for _ in range(len(hyperopt_parameters))]

        new_params = dict(initial_args)
        for k, state_i in zip(sorted(hyperopt_parameters.keys()), state):
            v = hyperopt_parameters[k]
            if isinstance(v, tuple):
                choice = v[state_i]
                if hasattr(choice, "item"):
                    choice = choice.item()
                new_params[k] = choice
            else:
                raise AssertionError("Only hyperopt_parameters with finite numbers of permutations can be used"
                                     "with __get_next_hyperopt_choice_generator__.")

        for i, key in enumerate(sorted(hyperopt_parameters.keys())):
            state[i] += 1
            if state[i] % len(hyperopt_parameters[key]) == 0:
                state[i] = 0
            else:
                break

        return new_params, state

    @staticmethod
    def print_run_results(args, hyperopt_parameters, run_index, score, run_time):
        message = "Hyperopt run [" + str(run_index) + "]:"
        best_params_message = ""
        for k in hyperopt_parameters:
            best_params_message += k + "=" + "{:.4f}".format(args[k]) + ", "
        best_params_message += "time={:.4f},".format(run_time) + "score={:.4f}".format(score)
        info(message, best_params_message)
        return best_params_message

    @staticmethod
    def print_metric_statistics(score_dicts):
        for key in score_dicts[0].keys():
            try:
                values = list(map(lambda x: x[key], score_dicts))
                info(key, "=", np.mean(values), "+-", np.std(values),
                     "CI=(", np.percentile(values, 2.5), ",", np.percentile(values, 97.5), "),",
                     "median=", np.median(values),
                     "min=", np.min(values),
                     "max=", np.max(values))
            except:
                error("Could not get key", key, "for all score dicts.")

    def run_hyperopt(self):
        import tensorflow.keras.backend as K

        files_to_copy = [
            "preprocessor_m.pickle", "preprocessor_x.pickle", "output_preprocessors.pickle",
            "args.pickle", "eval_score.pickle", "test_score.pickle", "calibrated.pickle", "losses.pickle",
            "train_predictions.tsv", "train_predictions.thresholded.tsv",
            "val_predictions.tsv", "val_predictions.thresholded.tsv",
            "test_predictions.tsv", "test_predictions.thresholded.tsv",
            "test_t.npy", "test_y_pred.npy", "test_y_true.npy",
            "threshold_t.npy", "threshold_y_pred.npy", "threshold_y_true.npy"
        ]

        num_hyperopt_runs = int(np.rint(self.args["num_hyperopt_runs"]))
        hyperopt_offset = int(np.rint(self.args["hyperopt_offset"]))
        hyperopt_metric_name = self.args["hyperopt_metric_name"]

        initial_args = dict(self.args)
        hyperopt_parameters = self.get_hyperopt_parameters()
        info("Performing hyperparameter optimisation with parameters:", hyperopt_parameters)

        state = None
        max_permutations = HyperparameterMixin.calculate_num_hyperparameter_permutations(hyperopt_parameters)
        max_num_hyperopt_runs = min(max_permutations, num_hyperopt_runs)  # Do not perform more runs than necessary.
        enumerate_all_permutations = max_permutations <= num_hyperopt_runs

        job_ids, score_dicts, test_score_dicts, eval_dicts = [], [], [], []
        for i in range(max_num_hyperopt_runs):
            run_start_time = time.time()

            hyperopt_parameters = self.get_hyperopt_parameters()
            if enumerate_all_permutations:
                self.args, state = HyperparameterMixin.get_next_hyperopt_choice(initial_args,
                                                                                hyperopt_parameters,
                                                                                state=state)
            else:
                self.args = HyperparameterMixin.get_random_hyperopt_parameters(initial_args,
                                                                               hyperopt_parameters,
                                                                               hyperopt_index=i)

            if i < hyperopt_offset:
                # Skip until we reached the hyperopt offset.
                continue

            resample_with_replacement = self.args["resample_with_replacement"]
            if resample_with_replacement:
                self.load_data()

            eval_set = "test"
            score_dict, test_dict = self.run_single(evaluate_against=eval_set)
            score_dicts.append(score_dict)
            test_score_dicts.append(test_dict)

            if self.args["hyperopt_against_eval_set"]:
                eval_dict = test_dict
            else:
                eval_dict = score_dict
            score = -eval_dict[hyperopt_metric_name]

            run_time = time.time() - run_start_time

            # This is necessary to avoid memory leaks when repeatedly building new models.
            K.clear_session()

            best_params_message = HyperparameterMixin.print_run_results(self.args,
                                                                        hyperopt_parameters,
                                                                        i, score, run_time)
            if score > self.best_score and self.args["do_train"]:
                self.best_score_index = i
                self.best_score = score
                self.best_params = best_params_message
                best_model_path = self.get_best_model_path()
                model_dir = os.path.dirname(best_model_path)
                model_name = os.path.basename(best_model_path)

                def copy_file(source_file_name, target_file_name):
                    source_file_path = join(model_dir, source_file_name)
                    target_file_path = join(model_dir, target_file_name)
                    if os.path.isfile(source_file_path):
                        shutil.copy(source_file_path, target_file_path)
                    elif os.path.isdir(source_file_path):
                        copy_tree(source_file_path, target_file_path)
                    else:
                        warn("Not moving {source_file_path:} to {target_file_path:} due to unexpected or missing file."
                             .format(source_file_path=source_file_path, target_file_path=target_file_path))

                # NOTE: The current model files will be overwritten at the next hyperoptimisation iteration.
                #       We retain the best model's config and parameters only.
                for file_to_copy in files_to_copy + [model_name]:
                    copy_file(file_to_copy, "best_" + file_to_copy)

        info("Best[", self.best_score_index, "] config was", self.best_params)
        self.args = initial_args

        info("Best_test_score:", test_score_dicts[self.best_score_index])

        best_model_path = self.get_best_model_path()
        model_dir = os.path.dirname(best_model_path)
        model_name = os.path.basename(best_model_path)

        def copy_and_remove_source_file(source_file_name, target_file_name):
            source_file_path = join(model_dir, source_file_name)
            target_file_path = join(model_dir, target_file_name)
            if os.path.isfile(source_file_path):
                shutil.copy(source_file_path, target_file_path)
                os.remove(source_file_path)
            elif os.path.isdir(source_file_path):
                copy_tree(source_file_path, target_file_path)
                shutil.rmtree(source_file_path)
            else:
                warn("Not moving {source_file_path:} to {target_file_path:} due to unexpected or missing file."
                     .format(source_file_path=source_file_path, target_file_path=target_file_path))

        for file_to_copy in files_to_copy + [model_name]:
            copy_and_remove_source_file("best_" + file_to_copy, file_to_copy)

        HyperparameterMixin.print_metric_statistics(score_dicts)

        # Override last score dicts with best.
        self.save_score_dicts(score_dicts[self.best_score_index],
                              test_score_dicts[self.best_score_index],
                              self.args["output_directory"])

        if len(score_dicts) != 0:
            return score_dicts[self.best_score_index]
