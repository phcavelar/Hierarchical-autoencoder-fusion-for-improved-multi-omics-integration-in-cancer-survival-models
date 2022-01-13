#!/bin/bash

bash setup.sh
bash recreate_data_and_splits.sh
bash reproduce_benchmarks.sh
bash reproduce_figures.sh
bash reproduce_tables.sh
