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
from ncore.data_access.base_data_loader import BaseDataLoader
from ncore.data_access.synthetic_outcomes.outcome_generator import OutcomeGenerator
from ncore.data_access.preprocessing import ImputeMissing, Standardise, DropMissing, OneHotDiscretise


class Europe2DataLoader(BaseDataLoader):
    def __init__(self, rdata_file_path, **kwargs):
        super(Europe2DataLoader, self).__init__(**kwargs)
        self.id_name = "id"
        self.age_name, self.sex_name, self.ethnicity_name, self.risk_group_name = \
            "age", "sex", "ethnicity", "risk"
        self.rdata_file_path = rdata_file_path
        self.mutation_names, self.drug_indicator_names, self.treatments_doses_names = None, None, None
        self.history_treatment_names = None

    def get_mutation_names(self):
        return self.mutation_names

    def get_drug_indicator_names(self):
        return self.drug_indicator_names

    def get_historic_treatments(self):
        return list(map(lambda xi: xi.replace("Hist_", ""), self.history_treatment_names))

    def transform_patients(self, rows):
        start_date_name, end_date_name = "OTHER_moddate", "OTHER_enddate"
        treatment_history, failure_w48 = "OTHER_treatment_history", "Out_w48"
        had_aids_name, days_to_failure_name = "aids", "Time to Failure (days)"
        base_vl_name, base_cd4_name = "VL", "CD4"  # Assuming these are pre-treatment variables.

        # Some treatments were not given in this study, but do appear in patient histories.
        historic_treatments = self.get_historic_treatments()
        drug_names = list(map(lambda xi: xi[4:], self.drug_indicator_names)) + historic_treatments
        drug_names_index_map = dict(zip(drug_names, range(len(drug_names))))

        for to_convert in [base_vl_name, base_cd4_name]:
            rows.loc[:, (to_convert,)] = pd.to_numeric(rows[to_convert], errors='coerce')

        outcome_names = [days_to_failure_name]
        p_id = np.squeeze(rows[[self.id_name]].values)
        y = np.squeeze(rows[outcome_names].values)
        covariates_names = [
            self.age_name, self.ethnicity_name, self.risk_group_name, had_aids_name, base_vl_name, base_cd4_name
        ]
        x = rows[covariates_names].values
        s = rows[self.treatments_doses_names].values
        t = rows[self.drug_indicator_names].values
        h = rows[self.history_treatment_names].values
        m = rows[self.mutation_names].values

        assert not np.any(np.isnan(y)) or not np.any(np.isinf(y)), "Outcomes must be present for all patients."

        return p_id, x, h, m, y, s, t, covariates_names, self.mutation_names, outcome_names

    def get_covariate_preprocessors(self, seed):
        max_num_elements_for_discretisation=10
        preprocessors = [
            DropMissing(max_fraction_missing=0.998),
            OneHotDiscretise(max_num_elements_for_discretisation=max_num_elements_for_discretisation),
            Standardise(max_num_elements_for_discretisation=max_num_elements_for_discretisation),
            ImputeMissing(random_state=seed, max_num_elements_for_discretisation=max_num_elements_for_discretisation),
        ]
        return preprocessors

    def get_output_preprocessors(self, seed):
        preprocessors = [
            DropMissing(max_fraction_missing=0.998),
            OneHotDiscretise(),
        ]
        return preprocessors

    def get_discrete_covariate_names_and_indices(self):
        covariate_names = ["ethnicity", "risk_group", "has_aids"]
        covariate_indices = [1, 2, 3]
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
        ]

    def get_patients(self):
        import pyreadr

        rows = pyreadr.read_r(self.rdata_file_path)["data"]

        self.mutation_names = rows.columns[1:69].values
        self.drug_indicator_names = rows.columns[76:94].values
        self.treatments_doses_names = rows.columns[94:112].values
        self.history_treatment_names = rows.columns[112:132].values

        return rows


class SemisyntheticEurope2DataLoader(OutcomeGenerator, Europe2DataLoader):
    def __init__(self, rdata_file_path, output_directory, max_num_centroids=1, do_plot=False,
                 combination_effect_ratio=0.2, combination_effect_mean=-0.03, combination_effect_std=0.015,
                 treatment_assignment_bias_coefficient=10, num_simulated_treatments=10):
        super(SemisyntheticEurope2DataLoader, self).__init__(
            rdata_file_path=rdata_file_path, output_directory=output_directory,
            num_simulated_treatments=num_simulated_treatments, max_num_centroids=max_num_centroids,
            do_plot=do_plot, combination_effect_ratio=combination_effect_ratio,
            combination_effect_mean=combination_effect_mean,
            combination_effect_std=combination_effect_std,
            treatment_assignment_bias_coefficient=treatment_assignment_bias_coefficient
        )
