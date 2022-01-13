#!/bin/bash

for f in ./src/reproduce_tables/*.R; do
  echo "$f"
  Rscript "$f" 
done

CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate thesis_lel

for f in ./src/reproduce_tables/*.py; do
  echo "$f"
  python "$f" 
done
