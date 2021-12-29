#!/usr/bin/env python
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
import pickle
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser
from ncore.apps.util import warn
from ncore.apps.visualisation.visualisation import invoke_r_script


def parse_parameters():
    parser = ArgumentParser(description='Prepares plots for the TCE paper.')
    parser.add_argument("--model_dir", default="",
                        help="Model directory where result data is stored.")
    parser.add_argument("--output_directory", default="",
                        help="Output directory where results plots are stored.")
    parser.add_argument("--experiment_id", default="1",
                        help="Experiment batch ID.")
    parser.add_argument("--metric_name", default="resampled.rmse",
                        help="Metric name to plot.")
    return vars(parser.parse_args())


class PreparePlotsApplication(object):
    def __init__(self, args):
        self.args = args
        output_directory = self.args["output_directory"]
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

    def get_file_path(self, method, dataset_name, app_shorthand="tce", is_eval=True):
        file_name = "{APP_SHORTHAND:}_{method:}_{DATASET_NAME:}_{EXPERIMENT_BATCH_ID:}{EVAL:}".format(
            APP_SHORTHAND=app_shorthand,
            method=method,
            DATASET_NAME=dataset_name,
            EXPERIMENT_BATCH_ID=self.args["experiment_id"],
            EVAL="_eval" if is_eval else ""
        )
        file_path = os.path.join(self.args["model_dir"], file_name)
        return file_path

    def get_eval_results(self, dataset_name, methods):
        metric_name = self.args["metric_name"]

        results = {}
        for method in methods:
            dname = self.get_file_path(method, dataset_name)
            fname = os.path.join(dname, "test_score.pickle")

            if not os.path.isfile(fname):
                warn("No results available at", fname)
                continue

            with open(fname, "rb") as fp:
                score = pickle.load(fp)

                results[method] = score[metric_name]
        return results

    def get_train_time(self, dataset_name, methods):
        import subprocess

        results = {}
        for method in methods:
            dname = self.get_file_path(method, dataset_name)
            fname = os.path.join(dname, "log.txt")
            process = subprocess.Popen(
                f"cat {fname} | grep 'train_model took' | awk '{{print $5}}'", stdout=subprocess.PIPE, shell=True
            )
            average_times = process.communicate()[0].decode("utf-8").strip()
            average_times = list(map(lambda x: float(x.strip()), average_times.strip().split("\n")))
            results[method] = average_times
        return results

    def get_eval_time(self, dataset_name, methods):
        import subprocess

        results = {}
        for method in methods:
            dname = self.get_file_path(method, dataset_name)
            fname = os.path.join(dname, "log.txt")
            process = subprocess.Popen(
                f"cat {fname} | grep 'evaluate_model took' | awk '{{print $5}}'", stdout=subprocess.PIPE, shell=True
            )
            average_times = process.communicate()[0].decode("utf-8").strip()
            average_times = list(map(lambda x: float(x.strip()), average_times.strip().split("\n")))
            results[method] = average_times
        return results

    def plot_overall_performance(self, plot_title_base="{dataset:}"):
        output_directory = self.args["output_directory"]
        for dataset in ["simulator-2000-5", "europe1", "europe2", "europe1synthetic", "europe2synthetic", "crispr3way"]:
            file_name = "{dataset:}_results.pdf".format(dataset=dataset)
            if dataset.startswith("crispr"):
                methods = [
                    "BalancedCounterfactualRelationEstimator", "CounterfactualRelationEstimatorNoMixing",
                    "Deconfounder", "GANITE", "GaussianProcess", "KNearestNeighbours", "LinearRegression", "TARNET"
                ]
            elif dataset.startswith("europe") and dataset.find("synthetic") == -1:
                methods = [
                    "BalancedCounterfactualRelationEstimator", "CounterfactualRelationEstimatorNoMixing",
                    "GaussianProcess", "KNearestNeighbours", "LinearRegression"
                ]
            else:
                methods = [
                    "BalancedCounterfactualRelationEstimator", "CounterfactualRelationEstimatorNoMixing",
                    "Deconfounder", "GANITE", "GaussianProcess", "KNearestNeighbours", "LinearRegression", "TARNET"
                ]

            y_axis_label = "RMSE"
            compare_from = "NCoRE\n(balanced)"
            if dataset == "europe1synthetic":
                plot_title = "Europe1"
                compare_to = "NCoRE"
            elif dataset == "europe2synthetic":
                plot_title = "Europe2"
                compare_to = "NCoRE"
            elif dataset == "europe1":
                plot_title = "Europe1 (factual)"
                compare_to = "NCoRE"
            elif dataset == "europe2":
                plot_title = "Europe2 (factual)"
                compare_to = "NCoRE"
            elif dataset == "simulator-2000-5":
                plot_title = "Simulated"
                compare_to = "NCoRE"
            elif dataset == "crispr3way":
                plot_title = "CRISPR Three-way Knockout"
                compare_to = "Deconfounder"
                y_axis_label = "pRMSE"
                compare_from = "NCoRE"
            else:
                plot_title = plot_title_base.format(dataset=dataset)
                compare_to = "kNN"

            results = self.get_eval_results(dataset, methods)
            invoke_r_script("performance_plot.R", args=[
                results, output_directory, file_name, plot_title, compare_to, y_axis_label, compare_from
            ], output_dir=output_directory, file_name=file_name, with_print=True)

    def plot_performance_by_num_treatments(self, plot_title="Number of Treatments"):
        output_directory = self.args["output_directory"]
        all_results = defaultdict(dict)
        for num_treatments in ["2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "14", "16"]:
            dataset = "simulator-2000-{num_treatments:d}".format(num_treatments=int(num_treatments))
            methods = [
                "BalancedCounterfactualRelationEstimator", "CounterfactualRelationEstimatorNoMixing", "GaussianProcess",
                "KNearestNeighbours", "LinearRegression"
            ]
            if int(num_treatments) <= 5:
                methods.append("TARNET")
            if int(num_treatments) <= 8:
                methods.append("Deconfounder")
            if int(num_treatments) <= 10:
                methods.append("GANITE")
            file_name = "num_treatments_results.pdf"
            results = self.get_eval_results(dataset, methods)
            for key in results.keys():
                all_results[key][num_treatments] = results[key]

        x_axis_name = "Number of Treatments k"
        skip_ganite = "false"
        with_legend = "false"
        invoke_r_script("performance_line_plot.R", args=[
            all_results, output_directory, file_name, plot_title, x_axis_name, skip_ganite, with_legend
        ], output_dir=output_directory, file_name=file_name, with_print=True)

    def plot_computational_performance_by_num_treatments(self, plot_title="Number of Treatments"):
        output_directory = self.args["output_directory"]
        all_results = defaultdict(dict)
        all_treatments = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "14", "16"]
        for num_treatments in all_treatments:
            dataset = "simulator-2000-{num_treatments:d}".format(num_treatments=int(num_treatments))
            methods = [
                "BalancedCounterfactualRelationEstimator", "CounterfactualRelationEstimatorNoMixing", "GaussianProcess",
                "KNearestNeighbours", "LinearRegression"
            ]
            if int(num_treatments) <= 5:
                methods.append("TARNET")
            if int(num_treatments) <= 8:
                methods.append("Deconfounder")
            if int(num_treatments) <= 10:
                methods.append("GANITE")
            results = self.get_eval_time(dataset, methods)
            results = self.get_train_time(dataset, methods)
            for key in results.keys():
                all_results[key][num_treatments] = results[key]

        file_name = "comp_performance_results.pdf"
        for key in all_results.keys():
            reference = 1  # all_results[key][all_treatments[0]]
            for t in all_treatments:
                if t in all_results[key]:
                    all_results[key][t] = (np.array(all_results[key][t]) / reference).tolist()

        x_axis_name = "Number of Treatments k"
        skip_ganite = "false"
        with_legend = "false"
        invoke_r_script("performance_line_plot.R", args=[
            all_results, output_directory, file_name, plot_title, x_axis_name, skip_ganite, with_legend
        ], output_dir=output_directory, file_name=file_name, with_print=True)

    def plot_performance_by_num_samples(self, plot_title="Number of Samples"):
        output_directory = self.args["output_directory"]
        all_results = defaultdict(dict)
        for num_samples in ["500", "750", "1000", "1250", "1500", "1750", "2000", "3000", "4000", "5000", "10000", "15000", "20000"]:
            dataset = "simulator-{num_samples:d}-5".format(num_samples=int(num_samples))
            methods = [
                "BalancedCounterfactualRelationEstimator", "CounterfactualRelationEstimatorNoMixing", "Deconfounder",
                "GANITE", "GaussianProcess", "KNearestNeighbours", "LinearRegression", "TARNET"
            ]
            file_name = "num_samples_results.pdf"
            results = self.get_eval_results(dataset, methods)
            for key in results.keys():
                all_results[key][num_samples] = results[key]

        x_axis_name = "Number of Samples n"
        skip_ganite = "true"
        with_legend = "false"
        invoke_r_script("performance_line_plot.R", args=[
            all_results, output_directory, file_name, plot_title, x_axis_name, skip_ganite, with_legend
        ], output_dir=output_directory, file_name=file_name, with_print=True)

    def plot_performance_by_treatment_assignment_bias(self, plot_title="Treatment Assignment Bias"):
        output_directory = self.args["output_directory"]
        all_results = defaultdict(dict)
        for treatment_assignment_bias in ["5", "7", "12", "15", "17", "20"]:
            dataset = "simulator-2000-5-{treatment_assignment_bias:d}".format(
                treatment_assignment_bias=int(treatment_assignment_bias)
            )
            methods = [
                "BalancedCounterfactualRelationEstimator", "CounterfactualRelationEstimatorNoMixing", "Deconfounder",
                "GANITE", "GaussianProcess", "KNearestNeighbours", "LinearRegression", "TARNET"
            ]
            file_name = "treatment_assignment_bias_results.pdf"
            results = self.get_eval_results(dataset, methods)
            for key in results.keys():
                all_results[key][treatment_assignment_bias] = results[key]

        x_axis_name = "Treatment Assignment Bias $\\kappa$"
        skip_ganite = "true"
        with_legend = "true"
        invoke_r_script("performance_line_plot.R", args=[
            all_results, output_directory, file_name, plot_title, x_axis_name, skip_ganite, with_legend
        ], output_dir=output_directory, file_name=file_name, with_print=True)

    def run(self):
        self.plot_overall_performance()
        self.plot_computational_performance_by_num_treatments()
        self.plot_performance_by_num_treatments()
        self.plot_performance_by_num_samples()
        self.plot_performance_by_treatment_assignment_bias()


if __name__ == '__main__':
    app = PreparePlotsApplication(parse_parameters())
    app.run()
