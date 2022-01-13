#!/bin/bash

for f in ./src/jobs/*.R; do
  Rscript "$f" 
done

CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate thesis

for f in ./src/jobs/*.py; do
  python "$f" 
done
