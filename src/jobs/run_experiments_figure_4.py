#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
import os
import sys

module_path = os.path.abspath(os.path.join("./src/"))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd
import torch
from model.autoencoders import (
    HierarchicalSAE,
    HierarchicalSAEEncodeOnly,
    HierarchicalSAEEncodeOnlyNet,
    HierarchicalSAENet,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skorch.callbacks import EarlyStopping, LRScheduler
from utils.utils import (
    FixRandomSeed,
    StratifiedSkorchSurvivalSplit,
    get_cka_similarity_overall,
    hierarchical_sae_criterion,
    hierarchical_sae_encode_only_criterion,
)


def main():
    with open("config/config.json") as f:
        config = json.load(f)
    for cancer in ["BLCA", "SARC"]:
        data = pd.read_csv(
            f"./data/processed/{cancer}/merged/{config['data_name_tcga']}"
        )
        for modalities in [7]:
            print(f"Starting: {cancer}")
            X = data[data.columns[2:]]
            X = X.loc[:, (X != X.iloc[0]).any()]
            y_str = data["OS"].astype(str) + "|" + data["OS.time"].astype(str)

            train_splits = pd.read_csv(
                f"./data/splits/{cancer}/{config['train_split_name_tcga']}"
            )
            test_splits = pd.read_csv(
                f"./data/splits/{cancer}/{config['test_split_name_tcga']}"
            )
            clinical_indices = [
                i for i in range(len(X.columns)) if "clinical" in X.columns[i]
            ]
            gex_indices = [
                i for i in range(len(X.columns)) if "gex" in X.columns[i]
            ]
            cnv_indices = [
                i for i in range(len(X.columns)) if "cnv" in X.columns[i]
            ]
            meth_indices = [
                i for i in range(len(X.columns)) if "meth" in X.columns[i]
            ]
            mirna_indices = [
                i for i in range(len(X.columns)) if "mirna" in X.columns[i]
            ]
            mut_indices = [
                i for i in range(len(X.columns)) if "mut" in X.columns[i]
            ]
            rppa_indices = [
                i for i in range(len(X.columns)) if "rppa" in X.columns[i]
            ]

            blocks = [
                clinical_indices,
                gex_indices,
                cnv_indices,
                meth_indices,
                mirna_indices,
                mut_indices,
                rppa_indices,
            ]
            # Make sure that all variables are considered in the blocks
            assert sum([len(i) for i in blocks]) == X.shape[1]

            for i in range(1):
                train_ix = train_splits.iloc[i, :].dropna().values
                test_ix = test_splits.iloc[i, :].dropna().values
                net = HierarchicalSAEEncodeOnlyNet(
                    module=HierarchicalSAEEncodeOnly,
                    criterion=hierarchical_sae_encode_only_criterion,
                    max_epochs=config["max_epochs"],
                    lr=config["lr"],
                    train_split=StratifiedSkorchSurvivalSplit(
                        10, stratified=True
                    ),
                    optimizer=torch.optim.Adam,
                    callbacks=[
                        ("seed", FixRandomSeed(config["seed"])),
                        (
                            "es",
                            EarlyStopping(
                                patience=10,
                                monitor="valid_loss",
                                load_best=True,
                            ),
                        ),
                        (
                            "sched",
                            LRScheduler(
                                torch.optim.lr_scheduler.ReduceLROnPlateau,
                                patience=3,
                                cooldown=5,
                                monitor="valid_loss",
                                verbose=False,
                            ),
                        ),
                    ],
                    verbose=0,
                    batch_size=-1,
                    module__blocks=blocks[:modalities],
                    module__lambda_q=0.001,
                    module__common_hidden_layers=0,
                )
                pipe = make_pipeline(StandardScaler(), net)

                pipe.fit(
                    X.iloc[train_ix, :].to_numpy().astype(np.float32),
                    y_str.iloc[train_ix].to_numpy().astype(str),
                )
                hierarchicalsae_encode_only_representations = pipe[
                    1
                ].module_.forward(
                    torch.tensor(
                        pipe[0].transform(
                            X.iloc[test_ix, :].to_numpy().astype(np.float32),
                        )
                    )
                )
                net = HierarchicalSAENet(
                    module=HierarchicalSAE,
                    criterion=hierarchical_sae_criterion,
                    max_epochs=config["max_epochs"],
                    lr=config["lr"],
                    train_split=StratifiedSkorchSurvivalSplit(
                        10, stratified=True
                    ),
                    optimizer=torch.optim.Adam,
                    callbacks=[
                        ("seed", FixRandomSeed(config["seed"])),
                        (
                            "es",
                            EarlyStopping(
                                patience=10,
                                monitor="valid_loss",
                                load_best=True,
                            ),
                        ),
                        (
                            "sched",
                            LRScheduler(
                                torch.optim.lr_scheduler.ReduceLROnPlateau,
                                patience=3,
                                cooldown=5,
                                monitor="valid_loss",
                                verbose=False,
                            ),
                        ),
                    ],
                    verbose=0,
                    batch_size=-1,
                    module__blocks=blocks,
                    module__lambda_q=0.001,
                    module__common_hidden_layers=0,
                )
                pipe = make_pipeline(StandardScaler(), net)

                pipe.fit(
                    X.iloc[train_ix, :].to_numpy().astype(np.float32),
                    y_str.iloc[train_ix].to_numpy().astype(str),
                )
                hierarchicalsae_representations = pipe[1].module_.forward(
                    torch.tensor(
                        pipe[0].transform(
                            X.iloc[test_ix, :].to_numpy().astype(np.float32),
                        )
                    )
                )

                hierarchicalsae_to_hierarchicalsae_encode_only_cka = (
                    get_cka_similarity_overall(
                        [
                            hierarchicalsae_representations[1][
                                :, [q for q in range(64 * i, (i + 1) * 64)]
                            ]
                            for i in range(7)
                        ]
                        + [hierarchicalsae_representations[-1]],
                        [
                            hierarchicalsae_encode_only_representations[-1][
                                :, [q for q in range(64 * i, (i + 1) * 64)]
                            ]
                            for i in range(7)
                        ]
                        + [hierarchicalsae_encode_only_representations[-2]],
                    )
                )
                pd.DataFrame(
                    hierarchicalsae_to_hierarchicalsae_encode_only_cka
                ).to_csv(
                    f"./data/cka/{cancer}/hierarchicalsaenet_hierarchicalsaenet_encode_only_cka_similarity.csv",
                    index=False,
                )


if __name__ == "__main__":
    sys.exit(main())
