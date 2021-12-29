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
import itertools
import numpy as np
from ncore.apps.util import info
from ncore.data_access.generator import get_last_row_id
from ncore.apps.util import convert_indicator_list_to_int
from ncore.apps.util import convert_int_to_indicator_list
from ncore.data_access.base_data_loader import BaseDataLoader


class ModelEvaluation(object):
    @staticmethod
    def combination_to_indicator_list(combination, num_treatments):
        indicator = np.zeros((num_treatments,))
        indicator[combination] = 1
        return indicator

    @staticmethod
    def get_selected_combinations(loader, max_degree=3):
        num_treatments = len(loader.drug_indicator_names)
        selection_set = list(set(range(num_treatments)))

        all_combinations = []
        for combinations_size in range(1, max_degree+1):
            all_combinations += list(itertools.combinations(selection_set, combinations_size))
        all_combinations = list(map(lambda x: ModelEvaluation.combination_to_indicator_list(list(x), num_treatments),
                                    all_combinations))
        all_combinations = list(map(lambda x: convert_indicator_list_to_int(x),
                                    all_combinations))
        return all_combinations

    @staticmethod
    def calculate_statistics_counterfactual(y_true, y_pred, t, selected_combinations, set_name, with_print):
        t = np.array([convert_indicator_list_to_int(ti) for ti in t])
        get_rmse = lambda y1, y2: np.sqrt(np.mean(np.square(y1 - y2)))
        get_mae = lambda y1, y2: np.mean(np.abs(y1 - y2))

        has_counterfactuals = len(y_true.shape) == 2 and y_true.shape[-1] > 1
        if has_counterfactuals:
            selected_combinations = list(selected_combinations)
            get_factual = lambda y: np.array([y[i, ti] for i, ti in enumerate(t)])

            rmse = get_rmse(y_true[:, selected_combinations], y_pred[:, selected_combinations])
            mae = get_mae(y_true[:, selected_combinations], y_pred[:, selected_combinations])

            factual_y_true = get_factual(y_true)
            factual_y_pred = get_factual(y_pred)
            factual_rmse = get_rmse(factual_y_true, factual_y_pred)
            factual_mae = get_mae(factual_y_true, factual_y_pred)

            ti_index_map = dict(zip(selected_combinations, np.arange(len(selected_combinations))))
            counterfactual_y_true = y_true[:, selected_combinations]
            counterfactual_y_pred = y_pred[:, selected_combinations]

            # NOTE: More explicit error calculations are necessary for counterfactuals,
            # since the factual treatment may not be part of __selected_combinations__ for all units.
            full_index_set = set(np.arange(len(selected_combinations)))
            all_diffs = []
            for i, ti in enumerate(t):
                if ti in ti_index_map:
                    counterfactual_idx_set = full_index_set - {ti_index_map[ti]}
                else:
                    counterfactual_idx_set = full_index_set
                diff = counterfactual_y_true[i, list(counterfactual_idx_set)] - \
                       counterfactual_y_pred[i, list(counterfactual_idx_set)]
                all_diffs += diff.tolist()

            counterfactual_rmse = np.sqrt(np.mean(np.square(all_diffs)))
            counterfactual_mae = np.mean(np.abs(all_diffs))

            if with_print:
                info("Performance on", set_name,
                     "RMSE =", rmse, ", fRMSE =", factual_rmse, ", cfRMSE =", counterfactual_rmse,
                     "MAE =", mae, ", fMAE =", factual_mae, ", cfMAE =", counterfactual_mae)
            return {
                "rmse": rmse,
                "frmse": factual_rmse,
                "cfrmse": counterfactual_rmse,
                "mae": mae,
                "fmae": factual_mae,
                "cfmae": counterfactual_mae,
            }
        else:
            if len(y_true.shape) == 2:
                y_true = y_true[:, 0]
            if len(y_pred.shape) == 2:
                y_pred = y_pred[:, 0]

            rmse = get_rmse(y_true, y_pred)
            mae = get_mae(y_true, y_pred)

            if with_print:
                info("Performance on", set_name,
                     "RMSE =", rmse, ", fRMSE =", rmse,
                     "MAE =", mae, ", fMAE =", mae)
            return {
                "rmse": rmse,
                "frmse": rmse,
                "mae": mae,
                "fmae": mae
            }

    @staticmethod
    def evaluate(model, generator, num_steps, set_name="Test set", output_preprocessors=None,
                 with_print=True, threshold=None, loader=None,
                 output_directory="", dataset="", max_degree=3):
        has_potential_outcomes = loader is not None and hasattr(loader, "potential_outcomes")
        num_treatments = len(loader.drug_indicator_names)
        if has_potential_outcomes:
            num_outcomes = list(loader.potential_outcomes.items())[0][1].shape[-1]

            if num_treatments > 10:
                # Limit combinations to those of __max_degrees__ for high __num_treatments__ (scalability).
                selected_combinations = ModelEvaluation.get_selected_combinations(loader, max_degree=max_degree)
            else:
                selected_combinations = np.arange(1, num_outcomes)  # Excl. t=0=no treatments applied.
            selected_combinations = set(selected_combinations)
        else:
            num_outcomes = 2**num_treatments
            selected_combinations = None

        def postprocess(y_pred):
            y_pred = np.array(
                BaseDataLoader.preprocess_variables(y_pred,
                                                    steps=output_preprocessors,
                                                    transform_fun_name="inverse_transform")[0]
            )
            return y_pred

        all_outputs, all_num_tasks = [], []
        for _ in range(num_steps):
            generator_outputs = next(generator)
            if len(generator_outputs) == 7:
                batch_x, batch_m, batch_h, batch_t, batch_s, labels_batch, sample_weight = generator_outputs
            else:
                batch_x, batch_m, batch_h, batch_t, batch_s, labels_batch = generator_outputs
            p_ids = get_last_row_id()
            y_pred = model.predict(batch_x, batch_m, batch_h, batch_t, batch_s)
            y_pred = postprocess(y_pred)
            if has_potential_outcomes:
                y_pred_base = np.zeros((len(labels_batch), num_outcomes))
                for this_idx, (y_predi, ti) in enumerate(zip(y_pred, batch_t)):
                    # Ensure factual ti is evaluated even if not in __selected_combinations__.
                    this_ti = convert_indicator_list_to_int(ti)
                    y_pred_base[this_idx, this_ti] = y_predi
                y_pred = y_pred_base

                for outcome_idx in range(1, num_outcomes):
                    if outcome_idx in selected_combinations:
                        indicator = convert_int_to_indicator_list(outcome_idx, min_length=int(np.log2(num_outcomes)))
                        cur_batch_t = np.repeat(np.expand_dims(indicator, axis=0), len(batch_t), axis=0)
                        cur_y_pred = model.predict(batch_x, batch_m, batch_h, cur_batch_t, batch_s)
                        cur_y_pred = postprocess(cur_y_pred)
                        y_pred[:, outcome_idx] = cur_y_pred[:, 0]

                labels_batch = np.array([loader.potential_outcomes[pid] for pid in p_ids])

                assert y_pred.shape == labels_batch.shape

                all_outputs.append((y_pred, labels_batch, batch_t))
            else:
                all_outputs.append((y_pred, labels_batch, batch_t))

        selected_slice = -1
        y_pred, y_true, t = [], [], []
        output_dim = model.output[0].shape[-1] if hasattr(model, "output") else 1
        for current_step in range(num_steps):
            model_outputs, labels_batch, batch_t = all_outputs[current_step]

            if isinstance(model_outputs, list):
                model_outputs = model_outputs[selected_slice]

            if isinstance(labels_batch, list):
                labels_batch = labels_batch[selected_slice]

            y_pred.append(model_outputs)
            y_true.append(labels_batch)
            t.append(batch_t)

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        t = np.concatenate(t, axis=0)

        if output_dim != 1:
            y_true = y_true.reshape((-1, output_dim))
            y_pred = y_pred.reshape((-1, output_dim))
        else:
            y_pred = np.squeeze(y_pred)

        if (y_true.ndim == 2 and y_true.shape[-1] == 1) and \
           (y_pred.ndim == 1 and y_pred.shape[0] == y_true.shape[0]):
           y_pred = np.expand_dims(y_pred, axis=-1)

        if len(y_true.shape) == 1 and len(y_pred.shape) == 2:
            y_pred = y_pred[:, 1]

        assert y_true.shape[-1] == y_pred.shape[-1]
        assert y_true.shape[0] == y_pred.shape[0]

        if output_directory != "":
            np.save(os.path.join(output_directory, "{}_y_pred.npy".format(set_name)), y_pred)
            np.save(os.path.join(output_directory, "{}_y_true.npy".format(set_name)), y_true)
            np.save(os.path.join(output_directory, "{}_t.npy".format(set_name)), t)

        score_dict = ModelEvaluation.calculate_statistics_counterfactual(y_true, y_pred, t, selected_combinations,
                                                                         set_name, with_print)
        return score_dict
