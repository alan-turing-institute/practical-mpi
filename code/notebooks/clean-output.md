# Notebook cleaning

The following can be used to clean the output from the notebooks

```
module purge
module load baskerville
module -q load bask-apps/live
module load JupyterLab/4.0.5-GCCcore-12.3.0

jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb
```
