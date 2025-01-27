# Practical MPI - Baskerville-Turing Training

[![LaTeX build](../../actions/workflows/pdflatex.yml/badge.svg)](../../actions/workflows/pdflatex.yml)
[![Slides](https://img.shields.io/badge/PDF-Slides-orange.svg?style=flat)](../gh-action-result/pdf-output/practical-mpi.pdf)

2025-01-30

MPI Session Part II: Converting GPT2 to use MPI and Lightning

## Details

This folder contains the slides for the MPI session of the Baskerville-Turing Training.

## Building the slides as PDF

Requirements:

1. Beamer packages
2. pdflatex

Build the PDF output using the included makefile:
```
make
```

The final output can be found as `practical-mpi.pdf`.

To clean out the intermediary build files and output files:
```
make clean
```

## Licence

BSD 2-clause; see the LICENSE file.

