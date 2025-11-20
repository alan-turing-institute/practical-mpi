# FineWeb processing

This directory contains code used for processing the FineWeb dataset.

https://huggingface.co/datasets/HuggingFaceFW/fineweb

A directory of pre-processed FineWeb sample-10BT will be provided, so there's no need to run this code.
It's only provided for reference and completeness.

Nevertheless, the steps below explain how to download and prepare the data in case this is needed.
The steps below should ideally be performed from a compute node.

## Setup

```sh
module purge
module load baskerville
module load bask-apps/live
module load Python/3.11.3-GCCcore-12.3.0

pushd 3-Training/code/fineweb
python3 -m venv venv
source ./venv/bin/activate
pip install --upgrade pip
pip install -r ../requirements.txt
```

## Download the data

```sh
python fineweb.py
```

The data will be downloaded to the `./edu_fineweb10B` directory.
In total there should be 100 files.
This should be comprised of 99 training files with names following the format `fineweb_train_%0dd.npy` and a single validation file called `fineweb_val_000000.npy`.

## Check the data

The `countfineweb.py` script will count the number of tokens in each of the training shards and give a total token count.

```sh
python countfineweb.py
```

Each shard (except the last) should be contain 100 000 000 tokens.
The total token count should be 9 853 989 344.
This doesn't include the validation tokens.

## Prepare the data

The `splitfineweb.py` script will split an input file into smaller shards.
The input file should be named `input.npy`.

```sh
python splitfineweb.py
```

By default this will accept an input file `input.npy` and output ten shards named with the format `fineweb_train_%0dd.npy`.

## Tidy up

```sh
deactivate
popd
```

The virtual environment directory `3-Training/code/fineweb/venv` and its contents can be safely deleted if it's no longer needed.

