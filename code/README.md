# GPT2 example code

This directory contains the practical code for running GPT2 on Baskerville.

You should take a look at this directory using JupyterLab, which we'll go through during the session.

Here's a brief explanation of the files and folders contained here:

| Filename                  | Description                                                 |
|:--------------------------|:------------------------------------------------------------|
| `train_gpt2.py`           | Base GPT2 implementation to build on top of                 |
| `train_gpt2_std.py`       | Standard GPT2 implementation with no parallelism            |
| `train_gpt2_ddp.py`       | GPT2 implementation using PyTorch and DDP                   |
| `train_gpt2_lit.py`       | GPT2 implementation using PyTorch Lightning                 |
| `requirements.txt`        | Requiremenets file used to create the Python virtualenv     |
| `activate.sh`             | Creates and activates the environment (modules, virtualenv) |
| `notebooks`               | Notebooks for generating the `diffs` for during the session |
| `batchfiles`              | SLURM batch files for running the example code              |
| `example-logs`            | Examples log files from previous runs                       |
| `fineweb`                 | Scripts for processing the FineWeb dataset                  |
| `solutions`               | Don't look in here!                                         |

Note that not all of these files may exist in the version you're provided with.
That's intentional.
The aim is to create the different implementations.
However you can get all of these files from the `devel` branch of this repository if you want to fast forward to them.

