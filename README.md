# NCoRE: Learning Counterfactual Representations for Treatment Combinations

![Code Coverage](https://img.shields.io/badge/Python-3.8-blue)

![NCoRE](http://schwabpatrick.com/img/ncore.png)

Neural Counterfactual Relation Estimation (NCoRE) is a deep-learning method for inferring conditional average
causal treatment effects (CATEs) for individual units from observational data specifically designed for settings in
which multiple treatments can be applied in combination. This repository contains source code to reproduce the 
experiments presented in [the published manuscript](https://arxiv.org/abs/2103.11175).

## Install

The latest release can be installed directly from Github (ideally into a virtual environment):

```bash
$ pip install -r requirements.txt
```

Installation should finish within minutes on standard hardware. This software has only been tested with Python 3.8.

## Datasets

CRISPR-KO data is from [Zhou et al. 2020](https://www.sciencedirect.com/science/article/pii/S2211124720310056).

Access to the EuResist HIV cohort data can be requested by bona-fide researchers
from the [EuResist network](https://www.euresist.org/).

## Use

The program can be initiated to compute results for training a combinatorial `$method` model (model names correspond to classes present in the `ncore.models.baselines` python package) on semi-synthetic data with outcomes simulated based on the `$dataset` dataset from the command line as follows:

```bash
$ python /path/to/ncore/apps/main.py  \
  --dataset="$dataset"  \
  --method="$method"  \
  --output_directory=/path/to/output_dir  \
  --do_train  \
  --do_hyperopt  \
  --do_evaluate
```

`$dataset` can be any value of `"crispr3way", "europe1synthetic", "europe2synthetic", "europe1", "europe2", "simulator"` 
to select the dataset to run against.

`$method` can be any value of `"BalancedCounterfactualRelationEstimator", "Deconfounder", "KNearestNeighbours", "CounterfactualRelationEstimator", "CounterfactualRelationEstimatorNoMixing", "GANITE", "GaussianProcess", "LinearRegression", "TARNET", "MTVAE"`

The available command line parameters are documented in [this source file](ncore/apps/parameters.py).

Training should finish in less than an hour on standard hardware (in 2021). 
After conclusion of the training, the outputs of the training process are written to `/path/to/output_dir` - 
including output predictions, a model binary, preprocessors, and training loss curves. 
In addition, relevant performance metric for different prediction horizons are written to stdout.

## Cite

Please consider citing, if you reference or use our methodology, code or results in your work:

    @article{parbhoo2021ncore,
      title={{NCoRE: Neural Counterfactual Representation Learning for Combinations of Treatments}},
      author={Parbhoo, Sonali and Bauer, Stefan and Schwab, Patrick},
      journal={arXiv preprint arXiv:2103.11175},
      year={2021}
    }

### License

[MIT License](LICENSE.txt)

### Authors

Patrick Schwab (GSK AIML), Sonali Parbhoo (Harvard University), Stefan Bauer (KTH Stockholm, GSK, CIFAR Azrieli Scholar) 

### Acknowledgements

PS is an employee and shareholder of GlaxoSmithKline plc. SP is supported by the Swiss National Science Foundation under P2BSP2_184359.