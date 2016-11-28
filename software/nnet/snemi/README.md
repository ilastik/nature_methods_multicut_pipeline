# Deploying ICv1

## Inference
* Make sure the dependencies are installed (you'll find them under `Antipasti/requirements.txt`).
* Modify inference configurations (under `Experiment/Configurations/*inferconf.yml`) or buy a machine with 4 GPUs.
* From the command-line, run `python Scripts/infer.py /path/to/configuration/file.yml`.

## Training
* Install dependencies, modify configurations (under `Experiment/Configurations/runconfigset.yml`)
* From the command-line, run `python Scripts/train.py /path/to/configuration/file.yml --device gpu0`, which should start training on gpu0. 
