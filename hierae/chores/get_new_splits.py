#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys

module_path = os.path.abspath(os.path.join("./src/"))
if module_path not in sys.path:
    sys.path.append(module_path)

import json
import random

import numpy as np
import pandas as pd
from utils.utils import RepeatedStratifiedSurvivalKFold


def main(cancer):
    with open("config/config.json") as f:
        config = json.load(f)
    data = pd.read_csv(
        f"./data/processed/{cancer}/merged/{config['data_name_tcga']}"
    )
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    # We don't care too much about the exact column choices, as really
    # only the target matters, because this script is only to create
    # the cross validation splits.
    X = data[data.columns[6:]]

    y_str = data["OS"].astype(str) + "|" + data["OS.time"].astype(str)
    cv = RepeatedStratifiedSurvivalKFold(
        n_repeats=config["outer_repetitions"],
        n_splits=config["outer_splits"],
        random_state=config["seed"],
    )
    splits = [i for i in cv.split(X, y_str)]
    pd.DataFrame([i[0] for i in splits]).to_csv(
        f"./data/splits/{cancer}/{config['train_split_name_tcga']}",
        index=False,
    )
    pd.DataFrame([i[1] for i in splits]).to_csv(
        f"./data/splits/{cancer}/{config['test_split_name_tcga']}",
        index=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
