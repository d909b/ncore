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
"""
import numpy as np
from torch.utils.data.sampler import BatchSampler
from ncore.apps.util import convert_indicator_list_to_int


class BalancedBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last, ball_trees, original_data, balancing_scores,
                 treatment_assignments, backlink_indices, seed=0):
        super(BalancedBatchSampler, self).__init__(sampler, batch_size, drop_last)
        self.ball_trees = ball_trees
        self.original_data = original_data
        self.backlink_indices = backlink_indices
        self.balancing_scores = balancing_scores
        self.treatment_assignments = treatment_assignments
        self.random_state = np.random.RandomState(seed)

    def __iter__(self):
        batch = set()
        step_offset, num_steps = 0, len(self)
        available_set = set(self.ball_trees.keys())
        for idx in self.sampler:
            batch.add(idx)

            if len(batch) == self.batch_size:
                yield batch
                batch = set()
                step_offset += 1
                if step_offset >= num_steps:
                    break
                else:
                    continue

            this_balancing_score = self.balancing_scores[idx]
            this_treatment_number = convert_indicator_list_to_int(self.treatment_assignments[idx])
            this_available_set = available_set - {this_treatment_number}

            emitted = False
            while len(this_available_set) != 0 and not emitted:
                choice = self.random_state.randint(len(this_available_set))
                ordered_choices = sorted(this_available_set)
                this_treatment_choice = ordered_choices[choice]

                current_max_k = self.ball_trees[this_treatment_choice].get_arrays()[0].shape[0]
                initial_k = min(current_max_k, 5)
                while True:
                    distance, sample_idx = self.ball_trees[this_treatment_choice]\
                        .query(this_balancing_score.reshape(1, -1), k=initial_k)

                    if sample_idx.shape[-1] < initial_k:
                        break

                    found_hit = False
                    for j in range(sample_idx.shape[-1]):
                        next_candidate_idx = sample_idx[0, j]
                        if next_candidate_idx not in batch:
                            dataset_idx = self.backlink_indices[this_treatment_choice][next_candidate_idx]
                            batch.add(dataset_idx)
                            if len(batch) == self.batch_size:
                                yield list(batch)
                                batch = set()
                                step_offset += 1
                                emitted = True
                            found_hit = True
                            break
                    if found_hit or initial_k == current_max_k:
                        break
                    initial_k = min(current_max_k, initial_k * 2)

                this_available_set -= {this_treatment_choice}

            if step_offset >= num_steps:
                break

    def __len__(self):
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore