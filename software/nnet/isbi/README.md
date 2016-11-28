# Deploying ICv1

Code for deploying ICv1 on ISBI data can be found under deploy as `deploy.py`. The configuration file can be found under `config/config.yaml`. Refer to the comments in there for further instructions. 

## Dependencies
The following dependencies must be installed (see also: `requirements.txt` under `Antipasti/requirements.txt`) and made available to the interpreter (e.g. by appending to `PYTHONPATH`):

* Theano
* h5py
* Vigra (required only for loading/saving .tiff data)
* PyYAML

Theano requires a `.theanorc` file in the user's `$HOME` directory. An example `.theanorc` file can be found under `config/.theanorc`.

## Obtaining Parameters
The pretrained parameters are available [here](http://hci.iwr.uni-heidelberg.de/system/files/private/downloads/1362115697/params.zip). Unzip the archive and provide path to the unzipped folder in the `config.yaml` file. 

## Running the Program
After having edited config.yaml and installed all dependencies, execute `deploy.py` with `python deploy.py`. If `mode` in `config.yaml` was set to 'infer', the results will be saved in `infpath` (also a flag in `config.yaml`). 

## Deploying on Your Own Data
To use ICv1 on your own data, refer to `deploy.py` and modify the source as required.

