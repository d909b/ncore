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
from pandas import read_csv
from itertools import chain
from ncore.data_access.base_data_loader import BaseDataLoader
from ncore.data_access.data_sources.base_data_source import BaseDataSource
from ncore.data_access.preprocessing import ImputeMissing, Standardise, DropMissing, OneHotDiscretise


class CRISPR3WayDataLoader(BaseDataLoader):
    """
    SOURCE: https://www.sciencedirect.com/science/article/pii/S2211124720310056
    CITE: Zhou et al. A Three-Way Combinatorial CRISPR Screen for Analyzing Interactions among Druggable Targets.
          Cell Reports 2020
    """
    def __init__(self, achilles_h5_file_path, csv_file_path, gene_list_file_path, **kwargs):
        super(CRISPR3WayDataLoader, self).__init__(**kwargs)
        self.achilles_h5_file_path = achilles_h5_file_path
        self.csv_file_path = csv_file_path
        self.gene_list_file_path = gene_list_file_path
        self.mutation_names = ["dummy_variable"]
        self.id_name = "genes"
        self.drug_indicator_names = self.get_gene_list()
        self.treatments_doses_names = ["Dose_{:s}".format(d) for d in self.drug_indicator_names]
        self.history_treatment_names = ["Hist_{:s}".format(d) for d in self.drug_indicator_names]
        self.log2_fold_change_name = "log2_fc"
        self.covariates_names = None

    def get_gene_list(self):
        rows = read_csv(self.csv_file_path).values
        gene_names = list(sorted(set(chain.from_iterable([row.split("_") for row in rows[:, 0].tolist()]))))
        return gene_names

    def get_mutation_names(self):
        return self.mutation_names

    def get_drug_indicator_names(self):
        return self.drug_indicator_names

    def get_historic_treatments(self):
        return self.history_treatment_names

    @staticmethod
    def map_history_to_treatment_indices(h, drug_names_index_map):
        new_h = []
        for i in range(len(h)):
            hi = h[i]
            this_hi = np.zeros((len(drug_names_index_map),))
            if hi is not None:
                for j in range(len(hi)):
                    hij = hi[j]
                    indices = list(map(lambda hijk: drug_names_index_map[hijk], hij.split("+")))
                    this_hi[indices] = 1
            new_h.append(this_hi)
        return np.array(new_h)

    def transform_patients(self, rows):
        historic_treatments = self.get_historic_treatments()
        drug_names = self.drug_indicator_names + historic_treatments

        p_id = np.squeeze(rows[[self.id_name]].values)
        outcome_names = [self.log2_fold_change_name]
        y = np.squeeze(rows[outcome_names].values)
        x = rows[self.covariates_names].values
        s = np.zeros((len(p_id), 1))
        t = rows[self.drug_indicator_names].values
        h = np.zeros((len(p_id), 1))
        m = np.zeros((len(p_id), 1))

        assert not np.any(np.isnan(y)) or not np.any(np.isinf(y)), "Outcomes must be present for all patients."

        return p_id, x, h, m, y, s, t, self.covariates_names, self.mutation_names, outcome_names

    def get_covariate_preprocessors(self, seed):
        max_num_elements_for_discretisation=10
        preprocessors = [
            DropMissing(max_fraction_missing=0.998),
            OneHotDiscretise(max_num_elements_for_discretisation=max_num_elements_for_discretisation),
        ]
        return preprocessors

    def get_output_preprocessors(self, seed):
        preprocessors = [
            DropMissing(max_fraction_missing=0.998),
            OneHotDiscretise(),
        ]
        return preprocessors

    def get_discrete_covariate_names_and_indices(self):
        covariate_names = []
        covariate_indices = []
        return covariate_names, covariate_indices

    def get_continuous_covariate_names_and_indices(self):
        covariate_names = []
        covariate_indices = []
        return covariate_names, covariate_indices

    def get_id_name(self):
        return self.id_name

    def get_split_property_names(self):
        return []

    def get_patients(self):
        achilles_data = BaseDataSource(self.achilles_h5_file_path)
        rows = read_csv(self.csv_file_path).values
        name_idx_map = dict(zip(self.drug_indicator_names, range(len(self.drug_indicator_names))))

        def get_covariates(gene_names):
            missing_covariates = np.zeros((achilles_data.get_shape()[-1],))
            covariate_set = []
            for i in range(3):
                covariates = missing_covariates
                if i < len(gene_names):
                    name = gene_names[i]
                    covariates_list = achilles_data.get_by_row_name(name)
                    if len(covariates_list) > 0:
                        covariates = covariates_list[0]
                covariate_set.append(covariates)
            covariate_set = np.concatenate(covariate_set)
            return covariate_set.tolist()

        new_rows = []
        for genes, a, b, c, ab, ac, bc, abc in rows:
            genes = genes.split("_")
            drug_indicators = np.zeros((len(self.drug_indicator_names,)))
            a_indicators = np.copy(drug_indicators)
            a_indicators[name_idx_map[genes[0]]] = 1
            b_indicators = np.copy(drug_indicators)
            b_indicators[name_idx_map[genes[1]]] = 1
            c_indicators = np.copy(drug_indicators)
            c_indicators[name_idx_map[genes[2]]] = 1

            new_rows.append((genes[0], a) + tuple(a_indicators.tolist()) + tuple(get_covariates(genes[0:1])))
            new_rows.append((genes[1], b) + tuple(b_indicators.tolist()) + tuple(get_covariates(genes[1:2])))
            new_rows.append((genes[2], c) + tuple(c_indicators.tolist()) + tuple(get_covariates(genes[2:])))
            new_rows.append((genes[0] + "_" + genes[1], ab) + tuple((a_indicators + b_indicators).tolist())
                            + tuple(get_covariates(genes[0:2])))
            new_rows.append((genes[0] + "_" + genes[2], ac) + tuple((a_indicators + c_indicators).tolist())
                            + tuple(get_covariates([genes[0], genes[2]])))
            new_rows.append((genes[1] + "_" + genes[2], bc) + tuple((b_indicators + c_indicators).tolist())
                            + tuple(get_covariates(genes[1:])))
            new_rows.append((genes[0] + "_" + genes[1] + "_" + genes[2], abc) +
                            tuple((a_indicators + b_indicators + c_indicators).tolist())
                            + tuple(get_covariates(genes)))
        self.covariates_names = [["{:s}_{:d}".format(name, i) for name in achilles_data.get_column_names()]
                                 for i in range(3)]
        self.covariates_names = list(chain.from_iterable(self.covariates_names))
        new_rows = pd.DataFrame(
            new_rows,
            columns=["genes", self.log2_fold_change_name] +
                    self.drug_indicator_names +
                    self.covariates_names
        )
        return new_rows
