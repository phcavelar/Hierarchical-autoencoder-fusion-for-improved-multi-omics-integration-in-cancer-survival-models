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
    ConcatSAE,
    ConcatSAENet,
    HierarchicalSAE,
    HierarchicalSAEEncodeOnly,
    HierarchicalSAEEncodeOnlyNet,
    HierarchicalSAENet,
    MeanSAEEncodeOnly,
    MeanSAEEncodeOnlyNet,
    PoolSAE,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skorch.callbacks import EarlyStopping, LRScheduler
from utils.utils import (
    FixRandomSeed,
    StratifiedSkorchSurvivalSplit,
    concat_sae_criterion,
    hierarchical_sae_criterion,
)


def main():
    with open("config/config.json") as f:
        config = json.load(f)
    modal_mapping = {
        7: "all_timed_euler",
        2: "[-rppa_cnv_meth_mirna_mut]_timed_euler",
    }
    for cancer in config["cancers"]:
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

            # HierarchicalSAE with extra hidden layer
            scores = []
            for i in range(train_splits.shape[0]):
                train_ix = train_splits.iloc[i, :].dropna().values
                test_ix = test_splits.iloc[i, :].dropna().values
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
                    module__blocks=blocks[:modalities],
                    module__lambda_q=0.001,
                    module__common_hidden_layers=1,
                )
                pipe = make_pipeline(StandardScaler(), net)

                pipe.fit(
                    X.iloc[train_ix, :].to_numpy().astype(np.float32),
                    y_str.iloc[train_ix].to_numpy().astype(str),
                )
                scores.append(
                    pipe.score(
                        X.iloc[test_ix, :].to_numpy().astype(np.float32),
                        y_str.iloc[test_ix].to_numpy().astype(str),
                    )
                )
            bench = pd.DataFrame()
            bench["concordance"] = scores
            bench["model"] = "hierarchicalsaenet"
            bench[["model", "concordance"]].to_csv(
                f"./data/benchmarks/{cancer}/hierarchicalsaenet_extra_hidden_layer_scores_{modal_mapping[modalities]}_es.csv",
                index=False,
            )

            # HierarchicalSAE without supervision
            scores = []
            for i in range(train_splits.shape[0]):
                train_ix = train_splits.iloc[i, :].dropna().values
                test_ix = test_splits.iloc[i, :].dropna().values
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
                    module__blocks=blocks[:modalities],
                    module__lambda_q=0.001,
                    module__common_hidden_layers=0,
                    module__supervise_aes=False,
                )
                pipe = make_pipeline(StandardScaler(), net)

                pipe.fit(
                    X.iloc[train_ix, :].to_numpy().astype(np.float32),
                    y_str.iloc[train_ix].to_numpy().astype(str),
                )
                scores.append(
                    pipe.score(
                        X.iloc[test_ix, :].to_numpy().astype(np.float32),
                        y_str.iloc[test_ix].to_numpy().astype(str),
                    )
                )
            bench = pd.DataFrame()
            bench["concordance"] = scores
            bench["model"] = "hierarchicalsaenet"
            bench[["model", "concordance"]].to_csv(
                f"./data/benchmarks/{cancer}/hierarchicalsaenet_scores_{modal_mapping[modalities]}_es_no_supervision.csv",
                index=False,
            )

            # ConcatSAE with extra hidden layer corresponds to
            # HierarchicalSAE (no decoder) with extra hidden layer.
            scores = []
            for i in range(train_splits.shape[0]):
                train_ix = train_splits.iloc[i, :].dropna().values
                test_ix = test_splits.iloc[i, :].dropna().values
                net = HierarchicalSAEEncodeOnlyNet(
                    module=HierarchicalSAEEncodeOnly,
                    criterion=concat_sae_criterion,
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
                    module__common_hidden_layers=1,
                )
                pipe = make_pipeline(StandardScaler(), net)

                pipe.fit(
                    X.iloc[train_ix, :].to_numpy().astype(np.float32),
                    y_str.iloc[train_ix].to_numpy().astype(str),
                )
                scores.append(
                    pipe.score(
                        X.iloc[test_ix, :].to_numpy().astype(np.float32),
                        y_str.iloc[test_ix].to_numpy().astype(str),
                    )
                )
            bench = pd.DataFrame()
            bench["concordance"] = scores
            bench["model"] = "hierarchicalsaenet_encode_only"
            bench[["model", "concordance"]].to_csv(
                f"./data/benchmarks/{cancer}/hierarchicalsaenet_encode_only_extra_hidden_layer_scores_{modal_mapping[modalities]}_es.csv",
                index=False,
            )

            # ConcatSAE without supervision
            scores = []
            for i in range(train_splits.shape[0]):
                train_ix = train_splits.iloc[i, :].dropna().values
                test_ix = test_splits.iloc[i, :].dropna().values
                net = ConcatSAENet(
                    module=ConcatSAE,
                    criterion=concat_sae_criterion,
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
                    module__supervise_aes=False,
                )
                pipe = make_pipeline(StandardScaler(), net)

                pipe.fit(
                    X.iloc[train_ix, :].to_numpy().astype(np.float32),
                    y_str.iloc[train_ix].to_numpy().astype(str),
                )
                scores.append(
                    pipe.score(
                        X.iloc[test_ix, :].to_numpy().astype(np.float32),
                        y_str.iloc[test_ix].to_numpy().astype(str),
                    )
                )
            bench = pd.DataFrame()
            bench["concordance"] = scores
            bench["model"] = "concatsaenet"
            bench[["model", "concordance"]].to_csv(
                f"./data/benchmarks/{cancer}/concatsaenet_scores_{modal_mapping[modalities]}_es_no_supervision.csv",
                index=False,
            )

            # MeanSAE with extra hidden layer
            scores = []
            for i in range(train_splits.shape[0]):
                train_ix = train_splits.iloc[i, :].dropna().values
                test_ix = test_splits.iloc[i, :].dropna().values
                net = MeanSAEEncodeOnlyNet(
                    module=MeanSAEEncodeOnly,
                    criterion=concat_sae_criterion,
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
                    module__common_hidden_layers=1,
                )
                pipe = make_pipeline(StandardScaler(), net)

                pipe.fit(
                    X.iloc[train_ix, :].to_numpy().astype(np.float32),
                    y_str.iloc[train_ix].to_numpy().astype(str),
                )
                scores.append(
                    pipe.score(
                        X.iloc[test_ix, :].to_numpy().astype(np.float32),
                        y_str.iloc[test_ix].to_numpy().astype(str),
                    )
                )
            bench = pd.DataFrame()
            bench["concordance"] = scores
            bench["model"] = "msaenet_encode_only"
            bench[["model", "concordance"]].to_csv(
                f"./data/benchmarks/{cancer}/msaenet_encode_only_scores_{modal_mapping[modalities]}_es.csv",
                index=False,
            )

            # MeanSAE without supervision
            scores = []
            for i in range(train_splits.shape[0]):
                train_ix = train_splits.iloc[i, :].dropna().values
                test_ix = test_splits.iloc[i, :].dropna().values
                net = ConcatSAENet(
                    module=PoolSAE,
                    criterion=concat_sae_criterion,
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
                    module__supervise_aes=False,
                )
                pipe = make_pipeline(StandardScaler(), net)

                pipe.fit(
                    X.iloc[train_ix, :].to_numpy().astype(np.float32),
                    y_str.iloc[train_ix].to_numpy().astype(str),
                )
                scores.append(
                    pipe.score(
                        X.iloc[test_ix, :].to_numpy().astype(np.float32),
                        y_str.iloc[test_ix].to_numpy().astype(str),
                    )
                )
            bench = pd.DataFrame()
            bench["concordance"] = scores
            bench["model"] = "msaenet"
            bench[["model", "concordance"]].to_csv(
                f"./data/benchmarks/{cancer}/msaenet_scores_{modal_mapping[modalities]}_es_no_supervision.csv",
                index=False,
            )

            # MaxSAE with extra hidden layer
            scores = []
            for i in range(train_splits.shape[0]):
                train_ix = train_splits.iloc[i, :].dropna().values
                test_ix = test_splits.iloc[i, :].dropna().values
                net = MeanSAEEncodeOnlyNet(
                    module=MeanSAEEncodeOnly,
                    criterion=concat_sae_criterion,
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
                    module__fusion="max",
                    module__common_hidden_layers=1,
                )
                pipe = make_pipeline(StandardScaler(), net)

                pipe.fit(
                    X.iloc[train_ix, :].to_numpy().astype(np.float32),
                    y_str.iloc[train_ix].to_numpy().astype(str),
                )
                scores.append(
                    pipe.score(
                        X.iloc[test_ix, :].to_numpy().astype(np.float32),
                        y_str.iloc[test_ix].to_numpy().astype(str),
                    )
                )
            bench = pd.DataFrame()
            bench["concordance"] = scores
            bench["model"] = "msaenet_max_encode_only"
            bench[["model", "concordance"]].to_csv(
                f"./data/benchmarks/{cancer}/msaenet_max_encode_only_scores_{modal_mapping[modalities]}_es.csv",
                index=False,
            )

            # MaxSAE without supervision
            scores = []
            for i in range(train_splits.shape[0]):
                train_ix = train_splits.iloc[i, :].dropna().values
                test_ix = test_splits.iloc[i, :].dropna().values
                net = ConcatSAENet(
                    module=PoolSAE,
                    criterion=concat_sae_criterion,
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
                    module__fusion="max",
                    module__supervise_aes=False,
                )
                pipe = make_pipeline(StandardScaler(), net)

                pipe.fit(
                    X.iloc[train_ix, :].to_numpy().astype(np.float32),
                    y_str.iloc[train_ix].to_numpy().astype(str),
                )
                scores.append(
                    pipe.score(
                        X.iloc[test_ix, :].to_numpy().astype(np.float32),
                        y_str.iloc[test_ix].to_numpy().astype(str),
                    )
                )
            bench = pd.DataFrame()
            bench["concordance"] = scores
            bench["model"] = "msaenet"
            bench[["model", "concordance"]].to_csv(
                f"./data/benchmarks/{cancer}/msaenet_scores_{modal_mapping[modalities]}_es_no_supervision.csv",
                index=False,
            )


if __name__ == "__main__":
    sys.exit(main())
