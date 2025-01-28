# SLURM batch files

This directory contains example batch files for use with SLURMfor running the code on Baskerville.

You may need to update the pathnames and the name of the python executable file to get these to run, but otherwise they should be self-contained.

Instructions for running them have been added as comments inside each file, but in practice it's always the same, something like this:

```
sbatch batch-1n-g.sh
```

| Batch file             | Description                                   |
|:-----------------------|:----------------------------------------------|
| `batch-1n1g.sh`        | Non-parallel 1 node 1 GPU                     |
| `batch-1n2g.sh`        | Non-parallel 1 node 2 GPUs (only uses 1 GPU)  |
| `batch-ddp-1n1g.sh`    | DDP 1 node 1 GPU (runs without parallism)     |
| `batch-ddp-1n2g.sh`    | DDP 1 node 2 GPUs                             |
| `batch-ddp-2n2g.sh`    | DDP 2 nodes 2 GPUs (one GPU per node)         |
| `batch-lit-1n1g.sh`    | Lightning 1 node 1 GPU                        |
| `batch-lit-1n2g.sh`    | Lightning 1 node 2 GPUs                       |
| `batch-lit-2n2g.sh`    | Lightning 2 nodes 2 GPUs (1 GPU per node)     |
| `batch-lit-2n8g.sh`    | Lightning 2 nodes 8 GPUs (4 GPUs per node)    |
| `batch-lit-jupyter.sh` | Lightning 1 node 1 GPU from inside JupyterLab |
| `batch-fineweb.sh`     | Run the FineWeb processing code               |

