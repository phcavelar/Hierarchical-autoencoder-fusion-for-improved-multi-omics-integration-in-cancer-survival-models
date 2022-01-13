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
from torch.nn.utils.prune import custom_from_mask
from utils.utils import (
    FixRandomSeed,
    StratifiedSkorchSurvivalSplit,
    get_max_blocks,
    get_max_blocks_cka,
    hierarchical_sae_criterion,
    hierarchical_sae_encode_only_criterion,
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

            model = "hierarchicalsaenet_encode_only"
            # HierarchicalSAE (no decoder) - CKA pruning
            scores_block_pruned = {i: [] for i in range(7)}
            for i in range(train_splits.shape[0]):

                print(f"Split: {i+1} / 10")
                train_ix = train_splits.iloc[i, :].dropna().values
                test_ix = test_splits.iloc[i, :].dropna().values
                net = HierarchicalSAEEncodeOnlyNet(
                    module=HierarchicalSAEEncodeOnly,
                    criterion=hierarchical_sae_encode_only_criterion,
                    max_epochs=100,
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
                prune_mask = torch.ones(
                    pipe[1].module_.hazard.hazard[1].weight.shape
                )
                tmp = pipe[1].module_.forward(
                    torch.tensor(
                        pipe[0].transform(
                            X.iloc[train_ix, :].to_numpy().astype(np.float32)
                        )
                    )
                )
                block_prune_order = (
                    get_max_blocks_cka(
                        [
                            tmp[-1][
                                :,
                                [
                                    z
                                    for z in range(
                                        i
                                        * pipe[
                                            1
                                        ].module_.block_embedding_dimension,
                                        (i + 1)
                                        * pipe[
                                            1
                                        ].module_.block_embedding_dimension,
                                    )
                                ],
                            ]
                            for i in range(len(blocks))
                        ],
                        tmp[-2],
                        7,
                        7,
                    )
                    .detach()
                    .numpy()[::-1][:-1]
                )
                scores_block_pruned[0].append(
                    pipe.score(
                        X.iloc[test_ix, :].to_numpy().astype(np.float32),
                        y_str.iloc[test_ix].to_numpy().astype(str),
                    )
                )
                for ix, block_to_prune in enumerate(block_prune_order):
                    prune_mask[
                        :,
                        [
                            q
                            for q in range(
                                block_to_prune
                                * pipe[1].module_.block_embedding_dimension,
                                (block_to_prune + 1)
                                * pipe[1].module_.block_embedding_dimension,
                            )
                        ],
                    ] = 0
                    custom_from_mask(
                        pipe[1].module_.encoder.encode[0],
                        "weight",
                        prune_mask,
                    )
                    scores_block_pruned[ix + 1].append(
                        pipe.score(
                            X.iloc[test_ix, :].to_numpy().astype(np.float32),
                            y_str.iloc[test_ix].to_numpy().astype(str),
                        )
                    )
            pd.DataFrame(scores_block_pruned).to_csv(
                f"./data/benchmarks/{cancer}/{model}_scores_block_pruned_cka_{modal_mapping[modalities]}_es.csv",
                index=False,
            )
            # HierarchicalSAE (no decoder) - magnitude pruning
            scores_block_pruned = {i: [] for i in range(7)}
            for i in range(train_splits.shape[0]):
                train_ix = train_splits.iloc[i, :].dropna().values
                test_ix = test_splits.iloc[i, :].dropna().values
                net = HierarchicalSAEEncodeOnlyNet(
                    module=HierarchicalSAEEncodeOnly,
                    criterion=hierarchical_sae_encode_only_criterion,
                    max_epochs=100,
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
                    module__lambda_1=0.001,
                    module__common_hidden_layers=0,
                )
                pipe = make_pipeline(StandardScaler(), net)

                pipe.fit(
                    X.iloc[train_ix, :].to_numpy().astype(np.float32),
                    y_str.iloc[train_ix].to_numpy().astype(str),
                )
                prune_mask = torch.ones(
                    pipe[1].module_.encoder.encode[0].weight.shape
                )
                block_prune_order = (
                    get_max_blocks(
                        pipe[1].module_.encoder.encode[0].weight, 7, 7
                    )
                    .detach()
                    .numpy()[::-1][:-1]
                )
                scores_block_pruned[0].append(
                    pipe.score(
                        X.iloc[test_ix, :].to_numpy().astype(np.float32),
                        y_str.iloc[test_ix].to_numpy().astype(str),
                    )
                )
                for ix, block_to_prune in enumerate(block_prune_order):
                    prune_mask[
                        :,
                        [
                            q
                            for q in range(
                                block_to_prune
                                * pipe[1].module_.block_embedding_dimension,
                                (block_to_prune + 1)
                                * pipe[1].module_.block_embedding_dimension,
                            )
                        ],
                    ] = 0
                    custom_from_mask(
                        pipe[1].module_.encoder.encode[0],
                        "weight",
                        prune_mask,
                    )
                    scores_block_pruned[ix + 1].append(
                        pipe.score(
                            X.iloc[test_ix, :].to_numpy().astype(np.float32),
                            y_str.iloc[test_ix].to_numpy().astype(str),
                        )
                    )
            pd.DataFrame(scores_block_pruned).to_csv(
                f"./data/benchmarks/{cancer}/{model}_scores_block_pruned_{modal_mapping[modalities]}_es.csv",
                index=False,
            )

            model = "hierarchicalsaenet"
            # HierarchicalSAE - CKA pruning
            scores_block_pruned = {i: [] for i in range(7)}
            for i in range(train_splits.shape[0]):
                train_ix = train_splits.iloc[i, :].dropna().values
                test_ix = test_splits.iloc[i, :].dropna().values
                net = HierarchicalSAENet(
                    module=HierarchicalSAE,
                    criterion=hierarchical_sae_criterion,
                    max_epochs=100,
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
                tmp = pipe[1].module_.forward(
                    torch.tensor(
                        pipe[0].transform(
                            X.iloc[train_ix, :].to_numpy().astype(np.float32)
                        )
                    )
                )
                prune_mask = torch.ones(
                    pipe[1].module_.ae.encode.encode[0].weight.shape
                )
                block_prune_order = (
                    get_max_blocks_cka(
                        [
                            tmp[1][:, [z for z in range(i * 64, (i + 1) * 64)]]
                            for i in range(len(blocks))
                        ],
                        tmp[-1],
                        7,
                        7,
                    )
                    .detach()
                    .numpy()[::-1][:-1]
                )
                scores_block_pruned[0].append(
                    pipe.score(
                        X.iloc[test_ix, :].to_numpy().astype(np.float32),
                        y_str.iloc[test_ix].to_numpy().astype(str),
                    )
                )
                for ix, block_to_prune in enumerate(block_prune_order):
                    prune_mask[
                        :,
                        [
                            q
                            for q in range(
                                block_to_prune
                                * pipe[1].module_.block_embedding_dimension,
                                (block_to_prune + 1)
                                * pipe[1].module_.block_embedding_dimension,
                            )
                        ],
                    ] = 0
                    custom_from_mask(
                        pipe[1].module_.ae.encode.encode[0],
                        "weight",
                        prune_mask,
                    )
                    scores_block_pruned[ix + 1].append(
                        pipe.score(
                            X.iloc[test_ix, :].to_numpy().astype(np.float32),
                            y_str.iloc[test_ix].to_numpy().astype(str),
                        )
                    )
            pd.DataFrame(scores_block_pruned).to_csv(
                f"./data/benchmarks/{cancer}/{model}_scores_block_pruned_cka_{modal_mapping[modalities]}_es.csv",
                index=False,
            )
            # HierarchicalSAE - magnitude pruning
            scores_block_pruned = {i: [] for i in range(7)}
            for i in range(train_splits.shape[0]):
                train_ix = train_splits.iloc[i, :].dropna().values
                test_ix = test_splits.iloc[i, :].dropna().values
                net = HierarchicalSAENet(
                    module=HierarchicalSAE,
                    criterion=hierarchical_sae_criterion,
                    max_epochs=100,
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
                prune_mask = torch.ones(
                    pipe[1].module_.ae.encode.encode[0].weight.shape
                )
                block_prune_order = (
                    get_max_blocks(
                        pipe[1].module_.ae.encode.encode[0].weight, 7, 7
                    )
                    .detach()
                    .numpy()[::-1][:-1]
                )
                scores_block_pruned[0].append(
                    pipe.score(
                        X.iloc[test_ix, :].to_numpy().astype(np.float32),
                        y_str.iloc[test_ix].to_numpy().astype(str),
                    )
                )
                for ix, block_to_prune in enumerate(block_prune_order):
                    prune_mask[
                        :,
                        [
                            q
                            for q in range(
                                block_to_prune
                                * pipe[1].module_.block_embedding_dimension,
                                (block_to_prune + 1)
                                * pipe[1].module_.block_embedding_dimension,
                            )
                        ],
                    ] = 0
                    custom_from_mask(
                        pipe[1].module_.ae.encode.encode[0],
                        "weight",
                        prune_mask,
                    )
                    scores_block_pruned[ix + 1].append(
                        pipe.score(
                            X.iloc[test_ix, :].to_numpy().astype(np.float32),
                            y_str.iloc[test_ix].to_numpy().astype(str),
                        )
                    )
            pd.DataFrame(scores_block_pruned).to_csv(
                f"./data/benchmarks/{cancer}/{model}_scores_block_pruned_{modal_mapping[modalities]}_es.csv",
                index=False,
            )


if __name__ == "__main__":
    sys.exit(main())
