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
import numpy as np
from scipy.stats import truncnorm
from sklearn.preprocessing import PolynomialFeatures
from ncore.data_access.meta_data.feature_types import FeatureTypeDiscrete
from ncore.apps.visualisation.treatment_effect_distributions import plot_treatment_effect_distributions
from ncore.apps.util import convert_indicator_list_to_int, convert_int_to_indicator_list, mixed_distance


class OutcomeGenerator(object):
    def __init__(self, output_directory, num_simulated_treatments=3, max_num_centroids=1, do_plot=False,
                 combination_effect_ratio=0.2, combination_effect_mean=-0.03, combination_effect_std=0.015,
                 treatment_assignment_bias_coefficient=10, **kwargs):
        super(OutcomeGenerator, self).__init__(**kwargs)
        self.initial_cd4_min, \
            self.initial_cd4_max, \
            self.initial_cd4_mean, \
            self.initial_cd4_std = 0.0, 3.14, 2.66, 0.47
        self.initial_vl_min, \
            self.initial_vl_max, \
            self.initial_vl_mean, \
            self.initial_vl_std = 0.84, 7.69, 4.48, 1.05
        self.potential_outcomes = {}
        self.centroids = None
        self.do_plot = do_plot
        self.output_directory = output_directory
        self.max_num_centroids = max_num_centroids
        self.num_simulated_treatments = num_simulated_treatments
        self.treatment_assignment_bias_coefficient = treatment_assignment_bias_coefficient

        # Ratio of combinations that have effect size > 0.
        self.combination_effect_ratio = combination_effect_ratio
        # Mean value of combination effect (normally distributed).
        self.combination_effect_mean = combination_effect_mean
        # Standard deviation of combination effect (normally distributed).
        self.combination_effect_std = combination_effect_std

    def sample_mixture_model(self, x, m, h, t, s, discrete_indices):
        rows = np.concatenate([x, m, h], axis=-1)
        num_centroids = np.random.randint(1, self.max_num_centroids+1)
        centroid_indices = np.random.permutation(np.arange(len(rows)))[:num_centroids]
        centroids = rows[centroid_indices]
        centroid_weights = np.random.rand(num_centroids)
        centroid_weights /= np.sum(centroid_weights)
        centroid_means = [self.initial_vl_min + (self.initial_vl_max - self.initial_vl_min)*np.random.rand()
                          for _ in range(len(centroids))]
        centroid_stds = [0.5]*len(centroids)
        centroid_mins = [self.initial_vl_min]*len(centroids)
        centroid_maxs = [self.initial_vl_max]*len(centroids)

        class MixtureModel(object):
            def __init__(self, centroids, centroid_weights,
                         centroid_means, centroid_stds, centroid_mins, centroid_maxs):
                self.centroids = centroids
                self.centroid_weights = centroid_weights
                self.centroid_means = centroid_means
                self.centroid_stds = centroid_stds
                self.centroid_mins = centroid_mins
                self.centroid_maxs = centroid_maxs
                assert len(self.centroids) == len(self.centroid_weights)
                assert len(self.centroids) == len(self.centroid_means)
                assert len(self.centroids) == len(self.centroid_stds)
                assert len(self.centroids) == len(self.centroid_mins)
                assert len(self.centroids) == len(self.centroid_maxs)

            def generate_outcome(self, x, m, h, t, s):
                concatenated_covariates = np.concatenate([x, m, h], axis=-1)
                selected_centroid = np.random.choice(np.arange(len(self.centroids)), p=self.centroid_weights)
                outcome = truncnorm.rvs(a=self.centroid_mins[selected_centroid],
                                        b=self.centroid_maxs[selected_centroid],
                                        loc=self.centroid_means[selected_centroid],
                                        scale=self.centroid_stds[selected_centroid])

                # Modulate by distance.
                distance = mixed_distance(concatenated_covariates, self.centroids[selected_centroid], discrete_indices)
                outcome *= distance
                return outcome

        return MixtureModel(centroids, centroid_weights, centroid_means, centroid_stds, centroid_mins, centroid_maxs)

    def generate_outcomes(self, y, p, x, m, h, s, t, x_feature_types,
                          x_train=None, m_train=None, h_train=None, s_train=None, t_train=None):
        self.drug_indicator_names, self.treatments_doses_names = \
            ["DRU_{:d}".format(d) for d in range(self.num_simulated_treatments)], \
            ["Dose_DRU_{:d}".format(d) for d in range(self.num_simulated_treatments)]

        if x_train is not None:
            discrete_indices = [idx for idx in range(len(x_feature_types))
                                if isinstance(x_feature_types[idx], FeatureTypeDiscrete)]
            self.mixture_models = [self.sample_mixture_model(x_train, m_train, h_train, t_train, s_train,
                                                             discrete_indices)
                                   for _ in range(self.num_simulated_treatments)]

        # Limit interactions to max. degree=5 to improve scalability of simulation.
        interactions = PolynomialFeatures(degree=min(self.num_simulated_treatments, 5),
                                          include_bias=True, interaction_only=True)  # Index 0 = bias = no treatment.
        interactions.fit_transform(np.zeros((1, self.num_simulated_treatments,)))

        interaction_degrees = np.log2(interactions.transform(np.ones((1, self.num_simulated_treatments,)) * 2))
        num_interactions = interaction_degrees.shape[-1]

        # Sparsify combination effects.
        combination_coefficients = (
                np.random.uniform(size=num_interactions) < self.combination_effect_ratio
        ).astype(float)
        combination_coefficients *= np.array([np.random.normal(loc=self.combination_effect_mean * (1.02**(degree-1)),
                                                               scale=self.combination_effect_std * (1.02**(degree-2)))
                                              for degree in interaction_degrees[0]])

        # Do not change base treatment effects (no interactions).
        combination_coefficients[:self.num_simulated_treatments+1] = 1.

        observed_indices = np.zeros((len(p),))
        num_treatment_combinations = 2 ** self.num_simulated_treatments
        potential_outcomes = np.zeros((len(p), self.num_simulated_treatments))
        new_t = np.zeros((len(p), self.num_simulated_treatments))
        for offset, (pi, xi, mi, hi, si, ti) in enumerate(zip(p, x, m, h, s, t)):
            potential_outcomes[offset] = np.array([self.mixture_models[t_idx].generate_outcome(xi, mi, hi, t_idx, si)
                                                   for t_idx in range(self.num_simulated_treatments)])

            # Ensure 0 < __observed_indices__ < __num_treatment_combinations__ (defined via num_simulated_treatments).
            observed_ti = max(convert_indicator_list_to_int(ti) % num_treatment_combinations, 1)
            observed_indices[offset] = observed_ti
            new_t[offset] = convert_int_to_indicator_list(observed_ti, min_length=self.num_simulated_treatments)

        treatment_effect_avg, treatment_effect_std = np.mean(potential_outcomes), np.std(potential_outcomes)

        # Standardise outcomes to zero mean and unit standard deviation.
        potential_outcomes -= treatment_effect_avg
        potential_outcomes /= treatment_effect_std

        potential_outcomes_w_interactions = interactions.transform(potential_outcomes.reshape((len(p), -1)))
        potential_outcomes_pre = (potential_outcomes_w_interactions * combination_coefficients)

        potential_outcomes = np.zeros((len(potential_outcomes_pre), num_treatment_combinations))
        for offset in np.arange(num_treatment_combinations):
            indicators = interactions.transform(np.reshape(
                convert_int_to_indicator_list(offset, min_length=self.num_simulated_treatments), (1, -1))
            )
            potential_outcomes[:, offset:offset+indicators.shape[0]] = np.dot(potential_outcomes_pre, indicators.T)

        potential_outcomes *= treatment_effect_std
        potential_outcomes += treatment_effect_avg

        self.potential_outcomes = {**self.potential_outcomes, **dict(zip(p, potential_outcomes))}
        y = np.array([potential_outcomes[offset, int(idx)] for offset, idx in enumerate(observed_indices)])

        if self.do_plot:
            plot_treatment_effect_distributions(potential_outcomes, self.output_directory)
        return y, new_t
