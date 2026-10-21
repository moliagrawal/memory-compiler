"""Trainer for Temporal Retrieval Network adhering to strict workspace rules."""

import os
import json
import time
import random
import subprocess
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from mc.models.temporal_net import TemporalRetrievalNet
from mc.models.losses import MultiObjectiveGroundingLoss
from mc.index.merge import generate_candidates_and_merge
from mc.eval.metrics import evaluate
from mc.eval.efficiency import get_gpu_memory_stats


def get_git_hash() -> str:
    """Retrieve current git commit hash."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res.stdout.strip() if res.returncode == 0 else "unknown_git_hash"
    except Exception:
        return "unknown_git_hash"


class GroundingFeatureDataset(Dataset):
    """
    In-memory dataset holding cached frame embeddings and query vectors.
    Since Charades features at 1fps are ~30 x 512 floats, all fit in RAM easily!
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        max_seq_len: int = 128,
        d_model: int = 512
    ):
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.d_model = d_model

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        frames = s["frames"]  # [T, D]
        query = s["query"]    # [D]
        start_t = s["start"]
        end_t = s["end"]
        timestamps = s["timestamps"]

        T = min(frames.shape[0], self.max_seq_len)
        frames_clipped = frames[:T]
        ts_clipped = timestamps[:T]

        # Padding
        pad_len = self.max_seq_len - T
        if pad_len > 0:
            pad_frames = np.zeros((pad_len, self.d_model), dtype=np.float32)
            padded_frames = np.concatenate([frames_clipped, pad_frames], axis=0)
            pad_mask = np.concatenate([np.ones(T, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)])
        else:
            padded_frames = frames_clipped
            pad_mask = np.ones(T, dtype=np.float32)

        # Ground truth labels
        gt_rel_mask = np.zeros(self.max_seq_len, dtype=np.float32)
        start_idx = 0
        end_idx = max(0, T - 1)

        for i, t in enumerate(ts_clipped):
            if start_t <= t <= end_t:
                gt_rel_mask[i] = 1.0

        if np.any(ts_clipped <= start_t):
            start_idx = int(np.where(ts_clipped <= start_t)[0][-1])
        if np.any(ts_clipped >= end_t):
            end_idx = int(np.where(ts_clipped >= end_t)[0][0])
        else:
            end_idx = T - 1

        end_idx = max(start_idx + 1, end_idx)
        start_idx = min(start_idx, self.max_seq_len - 1)
        end_idx = min(end_idx, self.max_seq_len - 1)

        return {
            "query_id": s["query_id"],
            "video_id": s["video_id"],
            "frames": torch.from_numpy(padded_frames).float(),
            "query": torch.from_numpy(query).float(),
            "mask": torch.from_numpy(pad_mask).float(),
            "gt_rel_mask": torch.from_numpy(gt_rel_mask).float(),
            "gt_start_idx": torch.tensor(start_idx, dtype=torch.long),
            "gt_end_idx": torch.tensor(end_idx, dtype=torch.long),
            "start_time": s["start"],
            "end_time": s["end"],
            "video_duration": s.get("duration", 30.0),
            "timestamps": torch.from_numpy(np.pad(ts_clipped, (0, pad_len))).float(),
            "valid_len": T
        }


def collate_grounding_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "query_ids": [b["query_id"] for b in batch],
        "video_ids": [b["video_id"] for b in batch],
        "frames": torch.stack([b["frames"] for b in batch]),
        "query": torch.stack([b["query"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "gt_rel_mask": torch.stack([b["gt_rel_mask"] for b in batch]),
        "gt_start_idx": torch.stack([b["gt_start_idx"] for b in batch]),
        "gt_end_idx": torch.stack([b["gt_end_idx"] for b in batch]),
        "start_times": [b["start_time"] for b in batch],
        "end_times": [b["end_time"] for b in batch],
        "video_durations": [b["video_duration"] for b in batch],
        "timestamps": torch.stack([b["timestamps"] for b in batch]),
        "valid_lens": [b["valid_len"] for b in batch]
    }


class TemporalNetTrainer:
    """
    Orchestrates training and validation for TemporalRetrievalNet.
    Outputs all artifacts required by Rule 1 & Rule 6 into `runs/<exp_id>/`.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        train_cfg = config.get("train", {})
        model_cfg = config.get("model", {})

        self.exp_id = train_cfg.get("exp_id", f"run_{int(time.time())}")
        self.run_dir = os.path.join("runs", self.exp_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self.seed = train_cfg.get("seed", 42)
        self.set_seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() and train_cfg.get("device") != "cpu" else "cpu")
        self.amp = train_cfg.get("amp", True) and (self.device.type == "cuda")

        self.model = TemporalRetrievalNet(
            d_in=model_cfg.get("d_in", 512),
            d_hidden=model_cfg.get("d_hidden", 256),
            num_heads=model_cfg.get("num_heads", 4),
            num_layers=model_cfg.get("num_layers", 3),
            dropout=model_cfg.get("dropout", 0.1),
            max_seq_len=model_cfg.get("max_seq_len", 128),
            cross_attention=model_cfg.get("cross_attention", False),
            use_transformer=model_cfg.get("use_transformer", True)
        ).to(self.device)

        weights = train_cfg.get("loss_weights", {})
        self.criterion = MultiObjectiveGroundingLoss(
            w_rel=weights.get("w_rel", 1.0),
            w_loc=weights.get("w_loc", 2.0),
            w_rank=weights.get("w_rank", 0.5)
        )

        self.lr = float(train_cfg.get("lr", 3e-4))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        self.log_file = os.path.join(self.run_dir, "log.txt")
        self.log(f"Initialized experiment: {self.exp_id} on {self.device}")

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def log(self, msg: str):
        print(f"[{self.exp_id}] {msg}")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            frames = batch["frames"].to(self.device)
            query = batch["query"].to(self.device)
            mask = batch["mask"].to(self.device)
            gt_rel = batch["gt_rel_mask"].to(self.device)
            gt_start = batch["gt_start_idx"].to(self.device)
            gt_end = batch["gt_end_idx"].to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.amp):
                out = self.model(frames, query, mask=mask)
                loss, loss_dict = self.criterion(
                    rel_logits=out["rel"],
                    start_logits=out["start"],
                    end_logits=out["end"],
                    gt_rel_mask=gt_rel,
                    gt_start_idx=gt_start,
                    gt_end_idx=gt_end,
                    pad_mask=mask
                )

            if self.amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return {"train_loss": round(total_loss / max(1, num_batches), 4)}

    @torch.inference_mode()
    def evaluate_dataset(self, samples: List[Dict[str, Any]]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        self.model.eval()
        preds = []
        gts = {}

        for s in samples:
            qid = s["query_id"]
            vid = s["video_id"]
            dur = s.get("duration", 30.0)
            gts[qid] = (s["start"], s["end"])

            frames = torch.from_numpy(s["frames"]).float().unsqueeze(0).to(self.device)
            query = torch.from_numpy(s["query"]).float().unsqueeze(0).to(self.device)
            timestamps = s["timestamps"]

            out = self.model(frames, query)
            rel_logits = out["rel"].squeeze(0).cpu().numpy()
            start_logits = out["start"].squeeze(0).cpu().numpy()
            end_logits = out["end"].squeeze(0).cpu().numpy()

            q_preds = generate_candidates_and_merge(
                rel_logits=rel_logits,
                start_logits=start_logits,
                end_logits=end_logits,
                timestamps=timestamps,
                video_duration=dur,
                query_id=qid,
                video_id=vid,
                final_top_k=5
            )
            preds.extend(q_preds)

        metrics = evaluate(preds, gts, tiou_thresholds=(0.3, 0.5, 0.7), top_k=(1, 5))
        return metrics, preds

    def fit(self, train_samples: List[Dict[str, Any]], val_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        train_cfg = self.config.get("train", {})
        epochs = train_cfg.get("max_epochs", 15)
        batch_size = train_cfg.get("batch_size", 64)
        patience = train_cfg.get("early_stop_patience", 4)

        train_ds = GroundingFeatureDataset(train_samples)
        train_loader = DataLoader(
            train_ds,
            batch_size=min(batch_size, len(train_ds)),
            shuffle=True,
            collate_fn=collate_grounding_batch,
            drop_last=False
        )

        best_score = -1.0
        best_metrics = {}
        best_preds = []
        patience_counter = 0

        self.log(f"Starting training: {epochs} epochs, {len(train_samples)} train samples, {len(val_samples)} val samples")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            loss_info = self.train_epoch(train_loader)
            val_metrics, val_preds = self.evaluate_dataset(val_samples)
            epoch_time = time.time() - t0

            val_target = val_metrics.get("R@1_tiou@0.5", 0.0)
            self.log(f"Epoch {epoch:02d}/{epochs:02d} - Loss: {loss_info['train_loss']} - Val R@1@0.5: {val_target:.4f} - mIoU: {val_metrics['mIoU']:.4f} - Time: {epoch_time:.2f}s")

            if val_target > best_score:
                best_score = val_target
                best_metrics = val_metrics
                best_preds = val_preds
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.run_dir, "best_checkpoint.pt"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.log(f"Early stopping triggered at epoch {epoch}")
                    break

        # Save final run artifacts required by Rule 1
        with open(os.path.join(self.run_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(self.config, f)

        # Remove internal top1 dict from serialized metrics
        clean_metrics = {k: v for k, v in best_metrics.items() if not k.startswith("_")}
        gpu_stats = get_gpu_memory_stats()
        clean_metrics["peak_vram_mb"] = gpu_stats["peak_vram_mb"]
        clean_metrics["vram_safe"] = gpu_stats["vram_limit_safe"]

        with open(os.path.join(self.run_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(clean_metrics, f, indent=2)

        with open(os.path.join(self.run_dir, "preds.jsonl"), "w", encoding="utf-8") as f:
            for p in best_preds:
                f.write(json.dumps(p) + "\n")

        with open(os.path.join(self.run_dir, "git_hash.txt"), "w", encoding="utf-8") as f:
            f.write(get_git_hash() + "\n")

        self.log(f"Run completed. Peak VRAM: {gpu_stats['peak_vram_mb']} MB (Safe: {gpu_stats['vram_limit_safe']})")
        return clean_metrics


def run_training_experiment(config_path: str) -> Dict[str, Any]:
    """CLI / programmatic helper to run experiment from config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    trainer = TemporalNetTrainer(config)
    return trainer.run_dir
