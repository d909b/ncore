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
import pandas as pd
from scipy.special import softmax
from scipy.stats import truncnorm
from tensorflow.keras.utils import to_categorical
from ncore.data_access.base_data_loader import BaseDataLoader
from ncore.apps.util import convert_indicator_list_to_int, mixed_distance
from ncore.data_access.synthetic_outcomes.outcome_generator import OutcomeGenerator
from ncore.data_access.preprocessing import ImputeMissing, Standardise, DropMissing, OneHotDiscretise


class SimulatorDataLoader(OutcomeGenerator, BaseDataLoader):
    def __init__(self, seed, output_directory, num_simulated_treatments=3, max_num_centroids=1, do_plot=False,
                 combination_effect_ratio=0.2, combination_effect_mean=-0.03, combination_effect_std=0.015,
                 treatment_assignment_bias_coefficient=10, num_simulated_patients=2000):
        super(SimulatorDataLoader, self).__init__(
            output_directory, num_simulated_treatments, max_num_centroids,
            do_plot, combination_effect_ratio, combination_effect_mean,
            combination_effect_std, treatment_assignment_bias_coefficient
        )
        self.had_aids_name, self.days_to_failure_name = "aids", "Time to Failure (days)"
        self.base_vl_name, self.base_cd4_name = "VL", "CD4"  # Assuming these are pre-treatment variables.
        self.id_name = "id"
        self.failure_w48 = "Out_w48"
        self.age_name, self.sex_name, self.ethnicity_name, self.risk_group_name = \
            "age", "sex", "ethnicity", "risk"
        self.seed = seed
        self.mutation_names = [
             'MU_RT41L', 'MU_RT62V', 'MU_RT65ENR', 'MU_RT67N', 'MU_RT69', 'MU_RT70ER'
           , 'MU_RT74V', 'MU_RT75I', 'MU_RT77L', 'MU_RT115F', 'MU_RT116Y', 'MU_RT151M'
           , 'MU_RT184VI', 'MU_RT210W', 'MU_RT215YF', 'MU_RT219QE', 'MU_RT90I', 'MU_RT98G'
           , 'MU_RT100I', 'MU_RT101EHP', 'MU_RT103NS', 'MU_RT106AIM', 'MU_RT108I'
           , 'MU_RT138AGKQR', 'MU_RT179DFLT', 'MU_RT181CIV', 'MU_RT188CLH', 'MU_RT190SA'
           , 'MU_RT221Y', 'MU_RT225H', 'MU_RT227C', 'MU_RT230IL', 'MU_RT236L'
           , 'MU_PR10IFVCR', 'MU_PR11I', 'MU_PR13V', 'MU_PR16E', 'MU_PR20RMITV', 'MU_PR24I'
           , 'MU_PR30N', 'MU_PR32I', 'MU_PR33IFV', 'MU_PR34Q', 'MU_PR35G', 'MU_PR36ILV'
           , 'MU_PR43T', 'MU_PR46IL', 'MU_PR47VA', 'MU_PR48V', 'MU_PR50LV', 'MU_PR53LY'
           , 'MU_PR54LVMTAS', 'MU_PR58E', 'MU_PR60E', 'MU_PR62V', 'MU_PR63P', 'MU_PR64LMV'
           , 'MU_PR69KR', 'MU_PR71VITL', 'MU_PR73CSTA', 'MU_PR74P', 'MU_PR76V', 'MU_PR77I'
           , 'MU_PR82ATFISL', 'MU_PR83D', 'MU_PR84V', 'MU_PR85V', 'MU_PR88DS', 'MU_PR89VIM'
           , 'MU_PR90M', 'MU_PR93LM'
        ]
        self.drug_indicator_names, self.treatments_doses_names = \
            ["DRU_{:d}".format(d) for d in range(self.num_simulated_treatments)], \
            ["Dose_DRU_{:d}".format(d) for d in range(self.num_simulated_treatments)]
        self.history_treatment_names = ["Hist_DRU_{:d}".format(d) for d in range(self.num_simulated_treatments)]
        assert len(self.drug_indicator_names) == self.num_simulated_treatments
        assert len(self.treatments_doses_names) == self.num_simulated_treatments
        assert len(self.history_treatment_names) == self.num_simulated_treatments

        self.column_names = [
            "id", "age", "sex", "ethnicity", "risk", "aids", "VL", "CD4",
        ] + self.mutation_names + self.drug_indicator_names + self.treatments_doses_names\
          + self.history_treatment_names + ["Out_w48"]
        self.num_simulated_patients = num_simulated_patients

    def get_mutation_names(self):
        return self.mutation_names

    def get_mutation_probabilities(self):
        mutation_p = [
            0.14074855034264627, 0.017923036373220874, 0.01845018450184502, 0.1117554032683184, 0.05745914602003163,
            0.09014232999472852, 0.023721665788086453, 0.011597258829731154, 0.010015814443858724, 0.00790722192936215,
            0.007380073800738007, 0.009488666315234581, 0.2124406958355298, 0.08434370057986294, 0.16025303110173958,
            0.06220347917764892, 0.04217185028993147, 0.0158144438587243, 0.014760147601476014, 0.018977332630469163,
            0.0764364786505008, 0.03479177648919346, 0.018977332630469163, 0.03900896151818661, 0.0279388508170796,
            0.028465998945703744, 0.010015814443858724, 0.02161307327358988, 0.017923036373220874, 0.007380073800738007,
            0.004217185028993147, 0.005798629414865577, 0.004217185028993147, 0.20980495519240908, 0.019504480759093307,
            0.25566684238270954, 0.0680021085925145, 0.09277807063784924, 0.023721665788086453, 0.04059040590405904,
            0.025303110173958882, 0.07590933052187665, 0.0158144438587243, 0.017923036373220874, 0.2477596204533474,
            0.02319451765946231, 0.07854507116499737, 0.02055877701634159, 0.02214022140221402, 0.014232999472851872,
            0.024775962045334738, 0.0764364786505008, 0.028465998945703744, 0.10648392198207696, 0.31259884027411705,
            0.6136004217185029, 0.29362150764364786, 0.03637322087506589, 0.20137058513442277, 0.03953610964681075,
            0.014760147601476014, 0.0158144438587243, 0.3437005798629415, 0.09383236689509752, 0.014232999472851872,
            0.042698998418555616, 0.02214022140221402, 0.03637322087506589, 0.048497627833421195, 0.09066947812335266,
            0.3716394306800211
        ]
        return mutation_p

    def get_drug_indicator_names(self):
        return self.drug_indicator_names

    def get_historic_treatments(self):
        return list(map(lambda xi: xi.replace("Hist_", ""), self.history_treatment_names))

    def get_covariate_names(self):
        covariates_names = [
            self.age_name, self.ethnicity_name, self.risk_group_name,
            self.had_aids_name, self.base_vl_name, self.base_cd4_name
        ]
        return covariates_names

    def transform_patients(self, rows):
        for to_convert in [self.base_vl_name, self.base_cd4_name]:
            rows.loc[:, (to_convert,)] = pd.to_numeric(rows[to_convert], errors='coerce')

        p_id = np.squeeze(rows[[self.id_name]].values)
        y = np.squeeze(rows[[self.failure_w48]].values)
        covariates_names = self.get_covariate_names()
        x = rows[covariates_names].values
        s = rows[self.treatments_doses_names].values
        t = rows[self.drug_indicator_names].values
        h = rows[self.history_treatment_names].values
        m = rows[self.mutation_names].values

        return p_id, x, h, m, y, s, t, covariates_names, self.mutation_names, [self.failure_w48]

    def get_covariate_preprocessors(self, seed):
        max_num_elements_for_discretisation = 10
        preprocessors = [
            DropMissing(max_fraction_missing=0.998),
            OneHotDiscretise(max_num_elements_for_discretisation=max_num_elements_for_discretisation),
            Standardise(max_num_elements_for_discretisation=max_num_elements_for_discretisation),
            ImputeMissing(random_state=seed, max_num_elements_for_discretisation=max_num_elements_for_discretisation),
        ]
        return preprocessors

    def get_output_preprocessors(self, seed):
        max_num_elements_for_discretisation = 10
        preprocessors = [
            DropMissing(max_fraction_missing=0.998),
            Standardise(max_num_elements_for_discretisation=max_num_elements_for_discretisation),
        ]
        return preprocessors

    def get_discrete_covariate_names_and_indices(self):
        covariate_names = ["ethnicity", "risk_group", "has_aids"] + self.drug_indicator_names
        covariate_indices = [1, 2, 3] + np.arange(-1, -len(self.drug_indicator_names)-1, -1).tolist()
        return covariate_names, covariate_indices

    def get_continuous_covariate_names_and_indices(self):
        covariate_names = ["age", "viral_load", "cd4"]
        covariate_indices = [0, 4, 5]
        return covariate_names, covariate_indices

    def get_id_name(self):
        return self.id_name

    def get_split_property_names(self):
        return [
            self.age_name,
            self.ethnicity_name,
            self.had_aids_name,
            self.risk_group_name,
        ] + self.drug_indicator_names

    def get_ethnicities(self):
        names = ["unknown", "other", "black", "asian", "white", "hispano-american"]
        p = np.array([1.93, 0.16, 0.86, 1.93, 91.96, 3.16]) / 100.
        return names, p

    def get_risk_groups(self):
        names = ["het", "idu", "others / unknown", "msm", "perinat", "blood"]
        p = np.array([23.96, 21.87, 2.83, 50.11, 0.37, 0.86]) / 100.
        return names, p

    def get_has_aids(self):
        names = [0, 1]
        p = np.array([76.2, 23.8]) / 100.
        return names, p

    def get_age(self, random_state):
        return float(random_state.randint(29, 53))

    def get_sex(self):
        names = [1, 2]
        p = np.array([80.07, 19.93]) / 100.
        return names, p

    def get_history(self):
        names = np.arange(self.num_simulated_treatments)
        p = np.random.uniform(0.05, 0.75, size=self.num_simulated_treatments)
        return names, p

    def get_actions(self):
        names = np.arange(self.num_simulated_treatments)
        p = np.random.uniform(0.05, 0.75, size=self.num_simulated_treatments)
        return names, p

    def get_initial_cd4(self):
        cd4 = truncnorm.rvs(a=self.initial_cd4_min,
                            b=self.initial_cd4_max,
                            loc=self.initial_cd4_mean,
                            scale=self.initial_cd4_std)
        return cd4

    def get_initial_vl(self):
        vl = truncnorm.rvs(a=self.initial_vl_min,
                           b=self.initial_vl_max,
                           loc=self.initial_vl_mean,
                           scale=self.initial_vl_std)
        return vl

    def simulate_treatment_data(self):
        random_state = np.random.RandomState(self.seed)
        action_names, action_p = self.get_actions()
        ethnicity_names, ethnicity_p = self.get_ethnicities()
        risk_names, risk_p = self.get_risk_groups()
        aids_names, aids_p = self.get_has_aids()
        sex_names, sex_p = self.get_sex()
        history_names, history_p = self.get_history()

        pre_rows, rows = [], []
        for p_id in range(self.num_simulated_patients):
            age = self.get_age(random_state)
            risk_group = random_state.choice(risk_names, p=risk_p)
            ethnicity = random_state.choice(ethnicity_names, p=ethnicity_p)
            has_aids = random_state.choice(aids_names, p=aids_p)
            sex = random_state.choice(sex_names, p=sex_p)
            history = (history_p > np.random.rand(len(history_p))).astype(int)
            assert len(history) == self.num_simulated_treatments

            pre_mutations = np.zeros((len(self.mutation_names),))
            for mutation_idx, mutation_probability in enumerate(self.get_mutation_probabilities()):
                this_mutation_p = [1. - mutation_probability, mutation_probability]
                pre_mutations[mutation_idx] = random_state.choice([0, 1], p=this_mutation_p)
            pre_cd4 = self.get_initial_cd4()
            log_pre_vl = self.get_initial_vl()

            pre_rows.append([p_id, age, sex, ethnicity, risk_group, has_aids, log_pre_vl, pre_cd4,
                             history, pre_mutations])

        ethnicities_map = dict(zip(ethnicity_names, np.arange(len(ethnicity_names))))
        risk_groups_map = dict(zip(risk_names, np.arange(len(risk_names))))

        def to_encoded_vector(age, sex, ethnicity, risk_group, has_aids, log_pre_vl, pre_cd4):
            return np.concatenate([
                [age, sex, log_pre_vl, pre_cd4, has_aids],
                to_categorical(ethnicities_map[ethnicity], num_classes=len(ethnicity_names)),
                to_categorical(risk_groups_map[risk_group], num_classes=len(risk_names)),
            ], axis=0)

        archetype_indices = np.random.permutation(np.arange(len(pre_rows)))[:self.num_simulated_treatments]
        archetypes = [pre_rows[idx] for idx in archetype_indices]
        archetypes = [to_encoded_vector(age, sex, ethnicity, risk_group, has_aids, log_pre_vl, pre_cd4)
                      for (_, age, sex, ethnicity, risk_group, has_aids, log_pre_vl, pre_cd4, _, _) in archetypes]
        discrete_indices = np.arange(
            len(archetypes[0]) - (len(ethnicity_names) + len(risk_names) + 1),
            len(archetypes[0])
        )
        for p_id, age, sex, ethnicity, risk_group, has_aids, log_pre_vl, pre_cd4, history, pre_mutations in pre_rows:
            demographics = [age, sex, ethnicity, risk_group, has_aids, log_pre_vl, pre_cd4]
            this_x = to_encoded_vector(age, sex, ethnicity, risk_group, has_aids, log_pre_vl, pre_cd4)

            num_assigned_treatments = min(np.random.poisson(2)+1, len(archetypes))
            num_selected = 0
            treatment_one_hot = np.zeros((len(archetypes),))
            while num_selected < num_assigned_treatments:  # Must have received some treatment.
                # Can be on more than one treatment.
                distances = np.array([mixed_distance(this_x, archetype, discrete_indices) for archetype in archetypes])
                distances[treatment_one_hot == 1.0] = 0.  # Remove previously selected treatments.

                drug_assignment_probabilities = softmax(self.treatment_assignment_bias_coefficient * distances)
                treatment_one_hot += (drug_assignment_probabilities > np.random.rand(len(action_p))).astype(int)
                treatment_one_hot[treatment_one_hot > 0] = 1
                treatment_one_hot = treatment_one_hot.astype(int)
                num_selected = sum(treatment_one_hot)

            potential_outcomes = np.zeros((2**len(action_names),))
            self.potential_outcomes[p_id] = potential_outcomes
            treatment_combination_offset = convert_indicator_list_to_int(treatment_one_hot.astype(int))
            row = [p_id] + demographics \
                  + pre_mutations.tolist() \
                  + treatment_one_hot.tolist() \
                  + treatment_one_hot.tolist() \
                  + history.tolist() \
                  + [potential_outcomes[treatment_combination_offset]]

            assert len(row) == len(self.column_names)
            rows.append(row)

        rows = pd.DataFrame(rows, columns=self.column_names)
        return rows

    def get_patients(self):
        rows = self.simulate_treatment_data()
        return rows
