# NCoRE: Learning Counterfactual Representations for Treatment Combinations

![Code Coverage](https://img.shields.io/badge/Python-3.8-blue)

## Datasets

TBD 

## Install

The latest release can be installed directly from Github (ideally into a virtual environment):

```bash
$ pip install .
```

Installation should finish within minutes on standard hardware. This software has been tested with Python 3.8.

## Use

The program can be initiated to compute results for training a combinatorial `LinearRegression` model (model names correspond to classes present in the `tce_dosage.models.baselines` python package) on semi-synthetic data with outcomes simulated based on the `Europe1` dataset from the command line as follows:

```bash
$ python3 /path/to/tce-dosage/apps/main.py 
  --dataset="simulator"
  --output_directory=/path/to/output_dir
  --do_train
  --do_evaluate
  --method=LinearRegression
  --num_simulated_treatments=3
```

Note that the number of simulated treatments can be changed using the command line setting `--num_simulated_treatmnets`. Training should finish within minutes on standard hardware. After conclusion of the training, the outputs of the training process are written to `/path/to/output_dir` - including output predictions, a model binary, preprocessors, and training loss curves. In addition, relevant performance metric for different prediction horizons are written to stdout.

## Cite

Please consider citing, if you reference or use our methodology, code or results in your work:

    @article{...,
        ...
    }

## Datasets

CRISPR-KO data is from https://www.sciencedirect.com/science/article/pii/S2211124720310056

### License

[MIT License](LICENSE.txt)

### Authors

Patrick Schwab, Sonali Parbhoo

### Acknowledgements

PS is an employee and shareholder of GlaxoSmithKline plc. SP is supported by the Swiss National Science Foundation under P2BSP2_184359.