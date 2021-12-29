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
import subprocess
import numpy as np
from os.path import join
from datetime import datetime
from abc import ABCMeta, abstractmethod
from ncore.apps.parameters import parse_parameters
from ncore.data_access.generator import make_generator
from ncore.apps.util import info, warn, get_num_available_gpus
from ncore.apps.hyperparameter_mixin import HyperparameterMixin


if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


@six.add_metaclass(ABCMeta)
class BaseApplication(HyperparameterMixin):
    def __init__(self, args):
        super(BaseApplication, self).__init__()
        self.args = args
        info("Args are:", self.args)
        version_string, has_uncommited_changes = self.get_code_version()
        info("Running version", version_string, "[UNCOMMITED CHANGES]" if has_uncommited_changes else "")
        info("Running at", str(datetime.now()))

        # Init output directory if it does not yet exist.
        os.makedirs(self.args["output_directory"], exist_ok=True)

        self.init_seeds()
        self.setup()

    def get_code_version(self):
        version_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./../../VERSION.txt")
        is_remote_run = os.path.isfile(version_file_path)  # VERSION.txt is present on remote runs.
        if is_remote_run:
            with open(version_file_path, "r") as fp:
                version_string = fp.read().strip()
            has_uncommited_changes = False  # Sync to remote fails if uncommitted changes are present.
        else:
            # Local runs use git directly to infer the code version.
            process = subprocess.Popen("git rev-parse HEAD", stdout=subprocess.PIPE, shell=True)
            version_string = process.communicate()[0].decode("utf-8").strip()
            process2 = subprocess.Popen("git diff-index --quiet HEAD -- || echo 'untracked'",
                                        stdout=subprocess.PIPE, shell=True)
            has_uncommited_changes = process2.communicate()[0].decode("utf-8").strip() == "untracked"
        return version_string, has_uncommited_changes

    def init_seeds(self):
        import random as rn
        import tensorflow.compat.v1 as tf

        seed = int(np.rint(self.args["seed"]))
        info("Seed is", seed)

        os.environ['PYTHONHASHSEED'] = '0'

        rn.seed(seed)
        np.random.seed(seed)

        tf.set_random_seed(seed)

    def setup(self):
        import tensorflow.compat.v1 as tf

        info("There are", get_num_available_gpus(), "GPUs available (TF version:", tf.__version__, ").")
        # tf.enable_eager_execution()
        # tf.disable_v2_behavior()
        tf.disable_eager_execution()

        #  Configure tensorflow not to use the entirety of GPU memory at start.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        import torch
        torch.manual_seed(0)
        torch.set_deterministic(True)
        torch.backends.cudnn.benchmark = False

        np.random.seed(0)

    @abstractmethod
    def train_model(self, train_generator, train_steps, val_generator, val_steps):
        return None, None

    @abstractmethod
    def evaluate_model(self, model, test_generator, test_steps, with_print=True, set_name="", threshold=None):
        return None

    @abstractmethod
    def make_train_generator(self):
        return None, None

    @abstractmethod
    def make_validation_generator(self):
        return None, None

    @abstractmethod
    def make_test_generator(self):
        return None, None

    @abstractmethod
    def get_best_model_path(self):
        return ""

    @abstractmethod
    def get_preprocessor_path_x(self):
        return ""

    @abstractmethod
    def get_preprocessor_path_m(self):
        return ""

    @abstractmethod
    def get_output_preprocessor_path(self):
        return ""

    @abstractmethod
    def get_prediction_path(self, set_name):
        return ""

    @abstractmethod
    def get_thresholded_prediction_path(self, set_name):
        return ""

    @abstractmethod
    def save_predictions(self, model, threshold=None):
        return

    @abstractmethod
    def load_data(self):
        return

    def run(self):
        if self.args["do_hyperopt"]:
            return self.run_hyperopt()
        else:
            evaluate_against = self.args["evaluate_against"]
            if evaluate_against not in ("test", "val"):
                warn("Specified wrong argument for --evaluate_against. Value was:", evaluate_against,
                     ". Defaulting to: val")
                evaluate_against = "val"
            return self.run_single(evaluate_against=evaluate_against)

    def run_single(self, evaluate_against="test"):
        info("Run with args:", self.args)

        save_predictions = self.args["save_predictions"]
        output_directory = self.args["output_directory"]
        with_test_uncertainty = self.args["with_test_uncertainty"]
        num_bootstrap_samples = int(np.rint(self.args["num_bootstrap_samples"]))

        train_generator, train_steps = self.make_train_generator()
        val_generator, val_steps = self.make_validation_generator()
        test_generator, test_steps = self.make_test_generator()

        info("Built generators with", train_steps,
             "training steps, ", val_steps,
             "validation steps and", test_steps, "test steps.")

        model, history = self.train_model(train_generator,
                                          train_steps,
                                          val_generator,
                                          val_steps)

        loss_file_path = join(output_directory, "losses.pickle")
        info("Saving loss history to", loss_file_path)
        pickle.dump(history, open(loss_file_path, "wb"), pickle.HIGHEST_PROTOCOL)

        args_file_path = join(self.args["output_directory"], "args.pickle")
        info("Saving args to", loss_file_path)
        pickle.dump(self.args, open(args_file_path, "wb"), pickle.HIGHEST_PROTOCOL)

        threshold = None
        if self.args["do_evaluate"]:
            if evaluate_against == "test":
                thres_generator, thres_steps = val_generator, val_steps
                eval_generator, eval_steps = test_generator, test_steps
                eval_set = self.test_set
            else:
                thres_generator, thres_steps = train_generator, train_steps
                eval_generator, eval_steps = val_generator, val_steps
                eval_set = self.validation_set

            # Get threshold from train or validation set.
            thres_score = self.evaluate_model(model, thres_generator, thres_steps,
                                              with_print=False, set_name="threshold")
            if "threshold" in thres_score:
                threshold = thres_score["threshold"]

            eval_score = self.evaluate_model(model, eval_generator, eval_steps,
                                             set_name=evaluate_against, threshold=threshold)

            if with_test_uncertainty:
                batch_size = int(np.rint(self.args["batch_size"]))
                num_losses = self.get_num_losses()

                resampled_scores, skip_index = [], 0
                for offset in range(num_bootstrap_samples):
                    while True:
                        resampled_generator, resampled_steps = make_generator(
                            dataset=eval_set,
                            batch_size=batch_size,
                            num_losses=num_losses,
                            shuffle=False,
                            resample_with_replacement=True,
                            seed=offset + skip_index
                        )
                        resampled_score = self.evaluate_model(
                            model, resampled_generator, resampled_steps,
                            set_name=evaluate_against + ".resample.{:d}".format(offset),
                            threshold=threshold
                        )
                        if resampled_score is not None:
                            resampled_scores.append(resampled_score)
                            # Re-run if bootstrap resampling yielded a population with only one of the target class.
                            break
                        skip_index += 1
                info("Bootstrap metrics ({:d} samples):".format(len(resampled_scores)))
                HyperparameterMixin.print_metric_statistics(resampled_scores)

                for key in resampled_scores[0].keys():
                    eval_score["resampled." + key] = [score[key] for score in resampled_scores]
        else:
            eval_score = None
            thres_score = None

        if save_predictions:
            self.save_predictions(model, threshold=threshold)

        if self.args["do_evaluate"]:
            if eval_score is None:
                test_score = self.evaluate_model(model, test_generator, test_steps,
                                                 with_print=evaluate_against == "val", set_name="test")
                eval_score = test_score
            else:
                test_score = eval_score
                eval_score = thres_score
        else:
            test_score = None

        BaseApplication.save_score_dicts(eval_score, test_score, self.args["output_directory"])
        return eval_score, test_score

    @staticmethod
    def save_score_dicts(eval_score, test_score, output_directory):
        eval_score_path = join(output_directory, "eval_score.pickle")
        with open(eval_score_path, "wb") as fp:
            pickle.dump(eval_score, fp, pickle.HIGHEST_PROTOCOL)
        test_score_path = join(output_directory, "test_score.pickle")
        with open(test_score_path, "wb") as fp:
            pickle.dump(test_score, fp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    app = BaseApplication(parse_parameters())
    app.run()
