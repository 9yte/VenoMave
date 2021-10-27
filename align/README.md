## General
Steps are based on this instructions: https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html

Download script to install Miniconda from here: https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh

## Install
install via:

```bash Miniconda3-latest-Linux-x86_64.sh
(yes to conda init)

## Create Env
create environment (we need python 3.8):

```conda create -n aligner -c conda-forge openblas python=3.8 openfst pynini ngram baumwelch

activate environment:

```conda activate aligner

run:

```pip install montreal-forced-aligner
```mfa thirdparty download
```mfa download acoustic english


## Run ALignment

```python src/data/align.py

if some error like this: "There was a problem locating libopenblas.so.0. Try installing openblas via system package manager?"

install on system
```sudo apt-get install libopenblas-dev
