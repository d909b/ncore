"""
Copyright (C) 2018  Patrick Schwab, ETH Zurich

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
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
from ncore.apps.util import info
from ncore.models.baselines.ganite_base.ganite_builder import GANITEBuilder


class GANITEModel(object):
    def __init__(self, input_dim, output_dim, num_units=128, dropout=0.0, l2_weight=0.0, learning_rate=0.0001,
                 num_layers=2, num_treatments=2, with_bn=False, nonlinearity="elu",
                 initializer=tf.variance_scaling_initializer(),
                 alpha=1.0, beta=1.0, seed=909):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.num_treatments = num_treatments
        self.seed = seed
        self.initial_predict_state = np.random.RandomState(seed)

        self.cf_generator_loss, self.cf_discriminator_loss, \
         self.ite_generator_loss, self.ite_discriminator_loss, \
          self.x, self.t, self.y_f, self.y_full, self.y_pred_cf, self.y_pred_ite, self.z_g, self.z_i = \
            GANITEBuilder.build(
                input_dim, output_dim,
                num_units=num_units,
                dropout=dropout,
                l2_weight=l2_weight,
                learning_rate=learning_rate,
                num_layers=num_layers,
                num_treatments=num_treatments,
                with_bn=with_bn,
                nonlinearity=nonlinearity,
                initializer=initializer,
                alpha=alpha,
                beta=beta
            )

    @staticmethod
    def get_scoped_variables(scope_name):
        t_vars = tf.trainable_variables()
        vars = [var for var in t_vars if scope_name in var.name]
        return vars

    @staticmethod
    def get_cf_generator_vairables():
        return GANITEModel.get_scoped_variables("g_cf")

    @staticmethod
    def get_cf_discriminator_vairables():
        return GANITEModel.get_scoped_variables("d_cf")

    @staticmethod
    def get_ite_generator_vairables():
        return GANITEModel.get_scoped_variables("g_ite")

    @staticmethod
    def get_ite_discriminator_vairables():
        return GANITEModel.get_scoped_variables("d_ite")

    def get_weight_variables(self):
        return GANITEModel.get_cf_generator_vairables() + GANITEModel.get_cf_discriminator_vairables() + \
               GANITEModel.get_ite_generator_vairables() + GANITEModel.get_ite_discriminator_vairables()

    def get_weights(self):
        weight_values = self.sess.run(self.get_weight_variables())
        return weight_values

    def set_weights(self, weights_values):
        ops = [weight.assign(weight_value) for weight, weight_value in zip(self.get_weight_variables(), weights_values)]
        self.sess.run(ops)

    def train(self, train_generator, train_steps, val_generator, val_steps, num_epochs,
              learning_rate, learning_rate_decay=0.97, iterations_per_decay=100,
              dropout=0.0, imbalance_loss_weight=0.0, l2_weight=0.0, checkpoint_path="",
              early_stopping_patience=12, early_stopping_on_pehe=False):
        global_step_1 = tf.Variable(0, trainable=False, dtype="int64")
        global_step_2 = tf.Variable(0, trainable=False, dtype="int64")
        global_step_3 = tf.Variable(0, trainable=False, dtype="int64")
        global_step_4 = tf.Variable(0, trainable=False, dtype="int64")

        opt = tf.train.AdamOptimizer(learning_rate)
        train_step_g_cf = opt.minimize(self.cf_generator_loss, global_step=global_step_1,
                                       var_list=GANITEModel.get_cf_generator_vairables())
        train_step_d_cf = opt.minimize(self.cf_discriminator_loss, global_step=global_step_2,
                                       var_list=GANITEModel.get_cf_discriminator_vairables())
        train_step_g_ite = opt.minimize(self.ite_generator_loss, global_step=global_step_3,
                                        var_list=GANITEModel.get_ite_generator_vairables())
        train_step_d_ite = opt.minimize(self.ite_discriminator_loss, global_step=global_step_4,
                                        var_list=GANITEModel.get_ite_discriminator_vairables())
        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        best_val_loss, num_epochs_without_improvement = np.finfo(float).max, 0
        for epoch_idx in range(num_epochs):
            for step_idx in range(train_steps):
                train_losses_g = self.run_generator(train_generator, 1, self.cf_generator_loss, train_step_g_cf)
                train_losses_d = self.run_generator(train_generator, 1, self.cf_discriminator_loss, train_step_d_cf)

            val_losses_g = self.run_generator(val_generator, val_steps, self.cf_generator_loss)
            val_losses_d = self.run_generator(val_generator, val_steps, self.cf_discriminator_loss)

            current_val_loss = val_losses_g[0]
            do_save = current_val_loss < best_val_loss
            if do_save:
                num_epochs_without_improvement = 0
                best_val_loss = current_val_loss
                saver.save(self.sess, checkpoint_path)
            else:
                num_epochs_without_improvement += 1

            self.print_losses(epoch_idx, num_epochs,
                              [train_losses_g[0], train_losses_d[0]],
                              [val_losses_g[0], val_losses_d[0]],
                              do_save)

            if num_epochs_without_improvement >= early_stopping_patience:
                break

        # Restore to best encountered.
        saver.restore(self.sess, checkpoint_path)

        best_val_loss, num_epochs_without_improvement = np.finfo(float).max, 0
        for epoch_idx in range(num_epochs):
            for step_idx in range(train_steps):
                train_losses_g = self.run_generator(train_generator, 1, self.ite_generator_loss, train_step_g_ite,
                                                    include_y_full=True)
                train_losses_d = self.run_generator(train_generator, 1, self.ite_discriminator_loss, train_step_d_ite,
                                                    include_y_full=True)
            val_losses_g = self.run_generator(val_generator, val_steps, self.ite_generator_loss,
                                              include_y_full=True)
            val_losses_d = self.run_generator(val_generator, val_steps, self.ite_discriminator_loss,
                                              include_y_full=True)

            current_val_loss = val_losses_g[0]
            do_save = current_val_loss < best_val_loss
            if do_save:
                num_epochs_without_improvement = 0
                best_val_loss = current_val_loss
                saver.save(self.sess, checkpoint_path)
            else:
                num_epochs_without_improvement += 1

            self.print_losses(epoch_idx, num_epochs,
                              [train_losses_g[0], train_losses_d[0]],
                              [val_losses_g[0], val_losses_d[0]],
                              do_save)

            if num_epochs_without_improvement >= early_stopping_patience:
                break

        # Restore to best encountered.
        saver.restore(self.sess, checkpoint_path)

    def print_losses(self, epoch_idx, num_epochs, train_losses, val_losses, did_save=False):
        info("Epoch [{:04d}/{:04d}] {:} TRAIN: G={:.3f} D={:.3f} VAL: G={:.3f} D={:.3f}"
            .format(
               epoch_idx, num_epochs,
               "xx" if did_save else "::",
               train_losses[0], train_losses[1],
               val_losses[0], val_losses[1]
            )
        )

    def run_generator(self, generator, steps, loss, train_step=None, include_y_full=False):
        losses = []
        for iter_idx in range(steps):
            (x_batch, t_batch), y_batch = next(generator)
            t_batch = np.expand_dims(t_batch, axis=-1)
            y_batch = np.expand_dims(y_batch, axis=-1)

            batch_size = len(x_batch)
            feed_dict = {
                self.x: x_batch,
                self.t: t_batch,
                self.y_f: y_batch,
                self.z_g: np.random.uniform(size=(batch_size, self.num_treatments-1)),
                self.z_i: np.random.uniform(size=(batch_size, self.num_treatments))
            }
            if include_y_full:
                y_pred = self._predict_g_cf([x_batch, t_batch], y_batch)
                y_pred[np.arange(len(y_pred)), t_batch] = y_batch
                feed_dict[self.y_full] = y_pred

            if train_step is not None:
                self.sess.run(train_step, feed_dict=feed_dict)

            losses.append(self.sess.run([loss],
                                        feed_dict=feed_dict))
        return np.mean(losses, axis=0)

    def _predict_g_cf(self, x, y_f):
        batch_size = len(x[0])
        y_pred = self.sess.run(self.y_pred_cf, feed_dict={
            self.x: x[0],
            self.t: x[1],
            self.y_f: y_f,
            self.z_g: np.random.uniform(size=(batch_size, self.num_treatments-1))
        })
        return y_pred

    def predict(self, x):
        batch_size = len(x[0])
        y_pred = self.sess.run(self.y_pred_ite, feed_dict={
            self.x: x[0],
            self.z_i: self.initial_predict_state.uniform(size=(batch_size, self.num_treatments))
        })
        y_pred = np.array(list(map(lambda inner, idx: inner[idx], y_pred, x[1])))
        return y_pred
