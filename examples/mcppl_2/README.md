# Multicut Pipeline V 2

We are currently working on updating this software to newer dependencies, improving the performance and also getting rid of legacy dependencies.
This version is not stable yet, we will update the main conda package to this version once it is.


## Installation

You can install this package via conda.
First, make sure that you have the ilastik and default channel listed in your ~/.condarc.
In addition, you need CPLEX with a valid license (we are working towards getting rid of this dependency).
Then, you can insall the package with

```
CPLEX_ROOT_DIR=/path/to/cplex conda create -n mc_ppl2 -c cpape multicut_pipeline
```

## Examples

See the scripts in this directory for basic examples on how to run the pipeline:

* init_exp.py: This initializes the cache for the experiments. It should be called only once.
If you need to change some of the input data (raw data, probability maps, over-segmentation, groundtruth),
you need to delete the old cache and call init_exp.py again.
* run_multicut.py: Run multicut. Trains a random forest on the training dataset and runs the multicut on the test dataset
with weights from random forest predictions.
* run_lifted_multicut.py: Runs the lifted multicut.
