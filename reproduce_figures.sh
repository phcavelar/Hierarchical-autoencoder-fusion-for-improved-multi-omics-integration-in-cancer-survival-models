#!/bin/bash

for f in ./src/reproduce_figures/*.R; do
  echo "$f"
  Rscript "$f" 
done