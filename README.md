# Hierarchical autoencoder fusion for improved multi-omics integration in cancer survival models

## Reproduction guide

To reproduce any of the results, all you have to do is run the requisite bash scripts.

You may obtain a copy the input all data files on [Kaggle](https://www.kaggle.com/davidwissel/hierarchicalautoencoderfusionformultiomics) for reproduction.

### Setup

`bash setup.sh`

Please note that you might have to run the setup script with `sudo` depending on your setup.

### Recreating data and splits (this is not necessary unless you want to, the preprocessed data is also included above)

Please note that to recreate the data and splits, you might need to adjust your Python version, which will be used in creating the splits. I attempted to make this as reproducible as possible, but unfortunately, the exact path can depend on your OS and conda version.

To change the version, first, run `which python` in your terminal after activating the requisite conda env via `conda activate hierarchical_fusion`. Afterward, change the following line in the `src/chores/get_new_cancers.R` script:

````R
Sys.setenv(PATH = paste(c(paste0("/Users/", Sys.info()[["user"]], "/miniforge3/envs/hierarchical_fusion/bin"), Sys.getenv("PATH"),
  collapse = .Platform$path.sep
), collapse = ":"))
``

to the following

```R
Sys.setenv(PATH = paste(c("your output from which python goes here", Sys.getenv("PATH"),
  collapse = .Platform$path.sep
), collapse = ":"))
````

Please note that this might also work by default if you use miniforge (i.e., you might not have to change it).

To recreate the data and splits then simply run `bash recreate_data_and_splits.sh`.

### Re-running benchmarks

To re-run all benchmarks, run `bash reproduce_benchmarks.sh`. Of course, you can also just re-run single scripts by manually calling the requisite `R` or `Python` script.

Please note that you need to re-run the benchmarks in order to reproduce figures and tables.

### Reproducing figures

`bash reproduce_figures.sh`

Figures can be reproduced without re-running any benchmarks, as the requisite benchmark results are included in the `data` folder obtainable from Kaggle as above.

### Reproducing tables

`bash reproduce_tables.sh`

Tables can be reproduced without re-running any benchmarks, as the requisite benchmark results are included in the `data` folder obtainable from Kaggle as above.

### Re-running everything

To re-run everything, simply call `bash reproduce_all.sh`. This will run in order the setup script, the data creation script, followed by the benchmarks, misc tasks, and the reproduction of all lots and tables.




In case of any problems, feel free to open an issue or email us at david.wissel@inf.ethz.ch.