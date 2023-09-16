# Learning the Multiple Traveling Salesmen Problem with Permutation Invariant Pooling Networks

This repository contains the code implementing the paper "[Learning the Multiple Traveling Salesmen Problem with Permutation Invariant Pooling Networks](https://arxiv.org/abs/1803.09621)".

## Getting Started

This project needs Python 3.6 and PyTorch 0.3.1 in order to run. To install the required packages into a new anaconda environment, run:

```
conda env create -f environment.yml
```

## Available Commands

All commands in this project reside under src/scripts, and can be run from the parent directory using:

```
python mTSP-DL command [args [..]]
```

### train

Used to train a model on mTSP-train. Useful flags include:

* `--train-paths`: The path to the training set.
* `--val-paths`: The path to the validation set.
* `--log-dir`: The directory into which to dump the Tensorboard logs.
* `--save-dir`: The directory into which to dump the checkpoint files.
* `--defaults-file` or `-d`: A configurations file, containing default values for the flags. For example:
```ini
[DEFAULTS]
train_paths = data/train/
val_paths = data/val/
log_dir = logs/
save_dir = checkpoints/
```

A possible useage can be:
```
python mTSP-DL train -d config.ini
```

### test_mtsp

Used to evaluate a trained model on mTSP-test. The arguments are:

* `weights`: The path to the saved checkpoint.
* `dataset`: The path to the test set.
* `beam_sizes`: A list of the beam sizes to use when evaluating the model.
* `-o` or `--other-args`: Other arguments to use when initializing the model.

### test_mtsplib

Used to evaluate a trained model on mTSPLib. The arguments are:

* `weights`: The path to the saved checkpoint.
* `tsplib_dir`: The path to the TSPLib directory.
* `output_file`: The path into which to output the results file.
* `meta`: The meta heuristic to use.
* `top`: The beam size of the beam search.
* `beam`: The number of solution checked, by the beam search and local search combined.
* `-o` or `--other-args`: Other arguments to use when initializing the model.

### test_tsp

Used to evaluate a trained model on a TSP dataset. The arguments are:

* `weights`: The path to the saved checkpoint.
* `dataset`: The path to the TSP dataset.
* `beam_sizes`: A list of the beam sizes to use when evaluating the model.
* `-o` or `--other-args`: Other arguments to use when initializing the model.
