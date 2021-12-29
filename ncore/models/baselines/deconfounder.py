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

ORIGINAL SOURCE: https://github.com/blei-lab/deconfounder_tutorial

MIT License

Copyright (c) 2018 Blei Lab

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
import sys
import json
import numpy as np
from scipy import sparse
import tensorflow.compat.v1 as tf
import tensorflow.keras.backend as K
from tensorflow_probability import edward2 as ed
from sklearn.linear_model import Ridge as LinearRegressionParent
from ncore.models.baselines.base_model import HyperparamMixin
from ncore.models.baselines.base_neural_network import BaseModel
from ncore.models.baselines.concatenate_composite_model import ConcatenateCompositeModel


if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


class Deconfounder(ConcatenateCompositeModel, HyperparamMixin):
    def __init__(self, form="linear", latent_dim=4, stddv_datapoints=0.1, max_num_iterations=50, num_treatments=3,
                 output_dim=None, missing_treatment_resolution_strategy="nearest", best_model_path=""):
        super(Deconfounder, self).__init__(
            num_treatments, output_dim, missing_treatment_resolution_strategy
        )
        self.model = self.log_joint = self.log_q = self.train = self.init = None
        self.params = None
        self.form = form
        self.learning_rate = 0.05
        self.latent_dim = latent_dim
        self.best_model_path = best_model_path
        self.stddv_datapoints = stddv_datapoints
        self.max_num_iterations = max_num_iterations
        self.max_num_iterations_inference = 25
        self.last_setup_datapoints = None

    @staticmethod
    def get_hyperparameter_ranges():
        ranges = {
            "latent_dim": (3, 5, 10, 15)
        }
        return ranges

    def _build_model(self, x_train, m_train, h_train, t_train, s_train, y_train):
        return LinearRegressionParent()

    def _get_holdout_mask(self, combined_x_train):
        num_datapoints, data_dim = combined_x_train.shape

        holdout_portion = 0.2
        n_holdout = int(holdout_portion * num_datapoints * data_dim)

        holdout_row = np.random.randint(num_datapoints, size=n_holdout)
        holdout_col = np.random.randint(data_dim, size=n_holdout)
        holdout_mask = sparse.coo_matrix(
            (np.ones(n_holdout), (holdout_row, holdout_col)),
            shape=combined_x_train.shape
        ).toarray()
        holdout_mask = np.minimum(1, holdout_mask)

        combined_x_train = np.multiply(1 - holdout_mask, combined_x_train)
        combined_x_val = np.multiply(holdout_mask, combined_x_train)
        return holdout_mask, combined_x_train, combined_x_val

    def _transform(self, data):
        holdout_mask, combined_x_train, combined_x_val = self._get_holdout_mask(data)
        num_datapoints, data_dim = combined_x_train.shape
        if self.last_setup_datapoints != num_datapoints:
            # Cache setup to avoid expensive recompilations.
            K.clear_session()
            self.setup(data_dim, num_datapoints, do_train=False, learning_rate=self.learning_rate)

        t = []
        with tf.Session() as sess:
            sess.run(self.init)
            sess.run(self.init_ops)

            for i in range(self.max_num_iterations_inference):
                sess.run(self.train, feed_dict={
                    self.data_x: combined_x_train,
                    self.holdout_mask: holdout_mask
                })
                if i % 5 == 0:
                    t.append(sess.run([self.elbo], feed_dict={
                        self.data_x: combined_x_train,
                        self.holdout_mask: holdout_mask
                    }))
                z_mean_inferred = sess.run(self.qz_mean)
        return z_mean_inferred

    def setup(self, data_dim, num_datapoints, do_train=True, learning_rate=0.05):
        if not do_train:
            self.last_setup_datapoints = num_datapoints
        else:
            self.last_setup_datapoints = None

        self.log_joint = ed.make_log_joint_fn(self.ppca_model)
        self.log_q = ed.make_log_joint_fn(self.variational_model)

        self.data_x = tf.placeholder(tf.float32, shape=[None, data_dim])
        self.holdout_mask = tf.placeholder(tf.float32, shape=[None, data_dim])
        self.qb_mean = tf.Variable(np.ones([1, data_dim]), dtype=tf.float32)
        self.qw_mean = tf.Variable(np.ones([self.latent_dim, data_dim]), dtype=tf.float32)
        self.qw2_mean = tf.Variable(np.ones([self.latent_dim, data_dim]), dtype=tf.float32)
        self.qz_mean = tf.Variable(np.ones([num_datapoints, self.latent_dim]), dtype=tf.float32)
        self.qb_stddv = tf.Variable(0 * np.ones([1, data_dim]), dtype=tf.float32)
        self.qw_stddv = tf.Variable(-4 * np.ones([self.latent_dim, data_dim]), dtype=tf.float32)
        self.qw2_stddv = tf.Variable(-4 * np.ones([self.latent_dim, data_dim]), dtype=tf.float32)
        self.qz_stddv = tf.Variable(-4 * np.ones([num_datapoints, self.latent_dim]), dtype=tf.float32)

        qb, qw, qw2, qz = self.variational_model(qb_mean=self.qb_mean, qb_stddv=tf.nn.softplus(self.qb_stddv),
                                                 qw_mean=self.qw_mean, qw_stddv=tf.nn.softplus(self.qw_stddv),
                                                 qw2_mean=self.qw2_mean, qw2_stddv=tf.nn.softplus(self.qw2_stddv),
                                                 qz_mean=self.qz_mean, qz_stddv=tf.nn.softplus(self.qz_stddv))

        energy = self.target(
            data_dim, num_datapoints, self.holdout_mask, self.data_x,
            qb, qw, qw2, qz
        )
        entropy = -self.target_q(
            self.qb_mean, tf.nn.softplus(self.qb_stddv),
            self.qw_mean, tf.nn.softplus(self.qw_stddv),
            self.qw2_mean, tf.nn.softplus(self.qw2_stddv),
            self.qz_mean, tf.nn.softplus(self.qz_stddv),
            qb, qw, qw2, qz
        )
        self.elbo = energy + entropy

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        if not do_train:
            param_vars = [self.qb_mean, self.qb_stddv, self.qw_mean, self.qw_stddv, self.qw2_mean, self.qw2_stddv]
            init_ops = [tf.assign(var, value) for var, value in zip(param_vars, self.params)]
            var_list = [self.qz_mean, self.qz_stddv]
            self.train = optimizer.minimize(-self.elbo, var_list=var_list)
            self.init = tf.global_variables_initializer()
            self.init_ops = init_ops
            return self.train, self.init, self.init_ops
        else:
            self.train = optimizer.minimize(-self.elbo)
            self.init = tf.global_variables_initializer()
            return self.train, self.init

    def fit(self, x_train, m_train, h_train, t_train, s_train, y_train,
                  x_val, m_val, h_val, t_val, s_val, y_val):
        combined_x_train = np.concatenate([x_train, m_train, h_train], axis=-1)
        combined_x_val = np.concatenate([x_val, m_val, h_val], axis=-1)
        holdout_mask, combined_x_train_masked, combined_x_val_masked = self._get_holdout_mask(combined_x_train)
        num_datapoints, data_dim = combined_x_train.shape
        train, init = self.setup(data_dim, num_datapoints)

        t = []
        with tf.Session() as sess:
            sess.run(init)

            for i in range(self.max_num_iterations):
                sess.run(train, feed_dict={
                    self.data_x: combined_x_train_masked,
                    self.holdout_mask: holdout_mask
                })
                if i % 5 == 0:
                    t.append(sess.run([self.elbo], feed_dict={
                        self.data_x: combined_x_train_masked,
                        self.holdout_mask: holdout_mask
                    }))

                b_mean_inferred = sess.run(self.qb_mean)
                b_stddv_inferred = sess.run(self.qb_stddv)
                w_mean_inferred = sess.run(self.qw_mean)
                w_stddv_inferred = sess.run(self.qw_stddv)
                w2_mean_inferred = sess.run(self.qw2_mean)
                w2_stddv_inferred = sess.run(self.qw2_stddv)
                z_mean_inferred = sess.run(self.qz_mean)
                z_stddv_inferred = sess.run(self.qz_stddv)

        self.params = [
            b_mean_inferred, b_stddv_inferred,
            w_mean_inferred, w_stddv_inferred,
            w2_mean_inferred, w2_stddv_inferred
        ]

        # TODO (@Sonali): Should the latents z be inferred with the masked or unmasked X?
        z_inferred_train = self._transform(combined_x_train)
        z_inferred_val = self._transform(combined_x_val)

        augmented_x_train = np.column_stack([x_train, z_inferred_train])
        augmented_x_val = np.column_stack([x_val, z_inferred_val])

        super(Deconfounder, self).fit(
            augmented_x_train, m_train, h_train, t_train, s_train, y_train,
            augmented_x_val, m_val, h_val, t_val, s_val, y_val
        )

    def predict(self, x, m, h, t, s):
        combined_x = np.concatenate([x, m, h], axis=-1)
        z_inferred = self._transform(combined_x)
        augmented_x = np.column_stack([x, z_inferred])
        y_pred = super(Deconfounder, self).predict(augmented_x, m, h, t, s)
        return y_pred

    def preprocess(self, x):
        return np.concatenate([x[0], np.atleast_2d(np.expand_dims(x[1], axis=-1))], axis=-1)

    def postprocess(self, y):
        if y.ndim > 1:
            return y[:, -1]
        else:
            return y

    def ppca_model(self, data_dim, latent_dim, num_datapoints, stddv_datapoints, mask):
        w = ed.Normal(loc=tf.zeros([latent_dim, data_dim]),
                      scale=tf.ones([latent_dim, data_dim]),
                      name="w")  # parameter
        z = ed.Normal(loc=tf.zeros([num_datapoints, latent_dim]),
                      scale=tf.ones([num_datapoints, latent_dim]),
                      name="z")  # local latent variable / substitute confounder
        if self.form == "linear":
            x = ed.Normal(loc=tf.multiply(tf.matmul(z, w), mask),
                          scale=stddv_datapoints * tf.ones([num_datapoints, data_dim]),
                          name="x")  # (modeled) data
        elif self.form == "quadratic":
            b = ed.Normal(loc=tf.zeros([1, data_dim]),
                          scale=tf.ones([1, data_dim]),
                          name="b")  # intercept
            w2 = ed.Normal(loc=tf.zeros([latent_dim, data_dim]),
                           scale=tf.ones([latent_dim, data_dim]),
                           name="w2")  # quadratic parameter
            x = ed.Normal(loc=tf.multiply(b + tf.matmul(z, w) + tf.matmul(tf.square(z), w2), mask),
                          scale=stddv_datapoints * tf.ones([num_datapoints, data_dim]),
                          name="x")  # (modeled) data
        return x, (w, z)

    def variational_model(self, qb_mean, qb_stddv, qw_mean, qw_stddv,
                          qw2_mean, qw2_stddv, qz_mean, qz_stddv):
        qb = ed.Normal(loc=qb_mean, scale=qb_stddv, name="qb")
        qw = ed.Normal(loc=qw_mean, scale=qw_stddv, name="qw")
        qw2 = ed.Normal(loc=qw2_mean, scale=qw2_stddv, name="qw2")
        qz = ed.Normal(loc=qz_mean, scale=qz_stddv, name="qz")
        return qb, qw, qw2, qz

    def target(self, data_dim, num_datapoints, holdout_mask, x_train, b, w, w2, z):
        """Unnormalized target density as a function of the parameters."""
        return self.log_joint(data_dim=data_dim,
                              latent_dim=self.latent_dim,
                              num_datapoints=num_datapoints,
                              stddv_datapoints=self.stddv_datapoints,
                              mask=1 - holdout_mask,
                              w=w, z=z, w2=w2, b=b, x=x_train)

    def target_q(self, qb_mean, qb_stddv, qw_mean, qw_stddv, qw2_mean, qw2_stddv,
                 qz_mean, qz_stddv, qb, qw, qw2, qz):
        return self.log_q(qb_mean=qb_mean, qb_stddv=qb_stddv,
                          qw_mean=qw_mean, qw_stddv=qw_stddv,
                          qw2_mean=qw2_mean, qw2_stddv=qw2_stddv,
                          qz_mean=qz_mean, qz_stddv=qz_stddv,
                          qw=qw, qz=qz, qw2=qw2, qb=qb)

    def get_config(self):
        config = {
            "form": self.form,
            "output_dim": self.output_dim,
            "latent_dim": self.latent_dim,
            "stddv_datapoints": self.stddv_datapoints,
            "max_num_iterations": self.max_num_iterations,
            "missing_treatment_resolution_strategy": self.missing_treatment_resolution_strategy,
            "best_model_path": self.best_model_path,
            "num_treatments": self.num_treatments,
        }
        return config

    @staticmethod
    def get_config_file_name():
        return "Deconfounder_config.json"

    @staticmethod
    def load(save_folder_path):
        config_file_name = Deconfounder.get_config_file_name()
        config_file_path = os.path.join(save_folder_path, config_file_name)
        with open(config_file_path, "r") as fp:
            config = json.load(fp)

        form = config["form"]
        latent_dim = config["latent_dim"]
        output_dim = config["output_dim"]
        stddv_datapoints = config["stddv_datapoints"]
        max_num_iterations = config["max_num_iterations"]
        missing_treatment_resolution_strategy = config["missing_treatment_resolution_strategy"]
        best_model_path = config["best_model_path"]
        num_treatments = config["num_treatments"]

        instance = Deconfounder(
            form=form,
            output_dim=output_dim,
            latent_dim=latent_dim,
            stddv_datapoints=stddv_datapoints,
            max_num_iterations=max_num_iterations,
            missing_treatment_resolution_strategy=missing_treatment_resolution_strategy,
            best_model_path=best_model_path,
            num_treatments=num_treatments
        )
        params = np.load(os.path.join(save_folder_path, Deconfounder.get_save_file_name()))
        num_params = len(params.files)
        params = [params[str(idx) + ".npy"] for idx in range(num_params)]
        instance.params = params
        with open(os.path.join(save_folder_path, Deconfounder.get_inner_save_file_name()), "rb") as load_file:
            instance.model = pickle.load(load_file)
        return instance

    def save(self, save_folder_path, overwrite=True):
        BaseModel.save_config(save_folder_path, self.get_config(), self.get_config_file_name(), overwrite, Deconfounder)
        params = dict([(str(i), weight) for i, weight in enumerate(self.params)])
        np.savez(os.path.join(save_folder_path, Deconfounder.get_save_file_name()), **params)
        with open(os.path.join(save_folder_path, Deconfounder.get_inner_save_file_name()), "wb") as save_file:
            pickle.dump(self.model, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def get_save_file_name():
        return "params.npz"

    @staticmethod
    def get_inner_save_file_name():
        return "inner_model.pickle"
