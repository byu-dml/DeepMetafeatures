| **Master**                                                                                                                                  | **Develop**                                                                                                                                 |
|----------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| [![Build Status](https://travis-ci.com/byu-dml/d3m-dynamic-neural-architecture.svg?branch=master)](https://travis-ci.com/byu-dml/d3m-dynamic-neural-architecture)                     | [![Build Status](https://travis-ci.com/byu-dml/d3m-dynamic-neural-architecture.svg?branch=develop)](https://travis-ci.com/byu-dml/d3m-dynamic-neural-architecture)                  |
| [![codecov](https://codecov.io/gh/byu-dml/d3m-dynamic-neural-architecture/branch/master/graph/badge.svg)](https://codecov.io/gh/byu-dml/d3m-dynamic-neural-architecture)  | [![codecov](https://codecov.io/gh/byu-dml/d3m-dynamic-neural-architecture/branch/develop/graph/badge.svg)](https://codecov.io/gh/byu-dml/d3m-dynamic-neural-architecture) |
# Dynamic Neural Architecture (DNA)

This repository contains a system for evaluating metalearning systems on a meta-dataset (a dataset containing info about machine learning experiments on datasets).
The meta-dataset was generated using the infrastructure created by Data Driven Discovery of Models (D3M), see https://gitlab.com/datadrivendiscovery.
Models range from the simple (random, linear regression) to the complex (deep neural networks, AutoSKLearn) and are evaluated on various metrics (see `dna/metrics.py`).

## Instructions for use:
0. Setup the python enviroment (`pip3 install -r requirements.txt`)
1. `main.sh` contains examples of how to run each metalearning model.
A more complete description of how to run a model can be found by running `python3 -m dna --help` or by inspecting `dna/__main__.py`.

## Configuration and results
1. Models are configured using JSON files, mapping function names to arguments like batch size, learning rate, and number of training epochs.
Examples can be found in `model_configs/<model name>_config.json`.
2. There are two meta-datasets available, `data/complete_classification.tar.xz` and `data/small_classification.tar.xz`.
The smaller is a subset of the larger for development purposes.
The first few lines of `main.sh` show how to use either dataset.
3. Complete results of running a metalearning model on the meta-dataset are written to the directory specified by the `--output-dir` flag and contain the arguments used to reproduce the results, model predictions, scores, plots, and any other model outputs such as model parameters.

## How to contribute a new model:
1. Add your new model code to `dna/models/_your_model_name_.py`.  It should inherit from the base classes of the tasks it can perform (RegressionBase, RankingBase, SubsetBase).
2. Please add tests to `tests/test_models.py`.
3. Once the model inherits from those classes and overrides their methods, the model should be imported and added to the list found in the function `get_models` of the file `dna/models.py`.
4. You can then run your model from the command line, or by adding it to `main.sh`

## How to contribute a new metric:
0. Add your metric code to `dna/metrics.py`.
1. Please add tests to the file `tests/test_metrics.py`.
2. The metrics are computed in `dna/problems.py`, in the appropriate problem class's `score` method, e.g. the Spearman Correlation Coefficient is computed with the `RankProblem`.
3. You can see your metric in action by running a model from the command line, or by running `main.sh`.
