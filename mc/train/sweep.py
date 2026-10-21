"""Hyperparameter grid search and ablation study execution (Phases 3 & 9)."""

import os
import copy
import yaml
import json
from typing import List, Dict, Any
from mc.train.trainer import TemporalNetTrainer


def generate_tuning_grid() -> List[Dict[str, Any]]:
    """
    Generate bounded tuning grid (Phase 3: max 24 runs):
      lr in {1e-4, 3e-4, 1e-3}
      layers in {2, 4}
      dropout in {0.1, 0.3}
      fusion in {hadamard_cat, cross_attn}
    """
    grid = []
    lrs = [1e-4, 3e-4, 1e-3]
    layers_list = [2, 4]
    dropouts = [0.1, 0.3]
    fusions = [False, True]  # cross_attention flag

    run_idx = 0
    for lr in lrs:
        for layers in layers_list:
            for dropout in dropouts:
                for cross_attn in fusions:
                    cfg = {
                        "exp_id": f"sweep_lr{lr}_L{layers}_d{dropout}_cross{cross_attn}",
                        "model": {
                            "d_in": 512,
                            "d_hidden": 256,
                            "num_heads": 4,
                            "num_layers": layers,
                            "dropout": dropout,
                            "cross_attention": cross_attn,
                        },
                        "train": {
                            "lr": lr,
                            "batch_size": 64,
                            "max_epochs": 10,
                            "early_stop_patience": 3,
                            "seed": 42
                        }
                    }
                    grid.append(cfg)
                    run_idx += 1
    return grid


def run_hyperparameter_sweep(
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    max_runs: int = 6
) -> Dict[str, Any]:
    """Execute hyperparameter sweep and freeze best config into configs/best.yaml."""
    grid = generate_tuning_grid()[:max_runs]
    best_exp = None
    best_score = -1.0
    best_cfg = None

    for idx, cfg in enumerate(grid, 1):
        print(f"\n--- Running Sweep Candidate {idx}/{len(grid)}: {cfg['exp_id']} ---")
        trainer = TemporalNetTrainer(cfg)
        metrics = trainer.fit(train_samples, val_samples)
        score = metrics.get("R@1_tiou@0.5", 0.0)

        if score > best_score:
            best_score = score
            best_exp = cfg["exp_id"]
            best_cfg = cfg

    # Freeze into configs/best.yaml
    if best_cfg:
        best_cfg_export = copy.deepcopy(best_cfg)
        best_cfg_export["train"]["exp_id"] = "best_frozen_model"
        with open("configs/best.yaml", "w", encoding="utf-8") as f:
            yaml.dump(best_cfg_export, f)

    return {
        "best_exp": best_exp,
        "best_r1_tiou05": best_score,
        "best_config": best_cfg
    }
