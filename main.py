"""
Main experiment entry point for the federated model watermarking framework.

Usage
─────
    python main.py                        # run with default config.py settings
    python main.py --rounds 10 --clients 5 --strategy badnets --dataset MNIST

All command-line arguments override the corresponding values in config.py.
"""

import argparse
import copy
import os
import sys

import torch
from torch.utils.data import DataLoader

import config
from data.dataset import (
    load_dataset,
    partition_iid,
    partition_non_iid_dirichlet,
    get_client_dataloaders,
    get_test_dataloader,
)
from models.cnn import build_model
from watermark.badnets import BadNetsWatermark
from watermark.pattern import PatternWatermark
from federated.client import Client
from federated.server import Server
from evaluation.metrics import evaluate_all
from utils.pruning import evaluate_pruning_robustness


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_watermark(strategy: str, cfg) -> object:
    """Instantiate the watermark object for the selected strategy."""
    strategy = strategy.lower()
    if strategy == "badnets":
        return BadNetsWatermark(
            target_label=cfg.WATERMARK_TARGET_LABEL,
            poison_rate=cfg.WATERMARK_POISON_RATE,
            trigger_size=cfg.TRIGGER_SIZE,
            trigger_value=cfg.TRIGGER_VALUE / 255.0,
        )
    if strategy == "pattern":
        return PatternWatermark(
            target_label=cfg.WATERMARK_TARGET_LABEL,
            poison_rate=cfg.WATERMARK_POISON_RATE,
            stride=cfg.PATTERN_STRIDE,
        )
    raise ValueError(f"Unknown watermark strategy: {strategy!r}. "
                     "Choose 'badnets' or 'pattern'.")


def build_clients(train_dataset, client_indices, watermark, cfg) -> list:
    """
    Build one :class:`Client` per partition.

    Clients whose index is in ``cfg.WATERMARK_CLIENTS`` receive a poisoned
    version of their local dataset.
    """
    clients = []
    for idx, indices in enumerate(client_indices):
        from torch.utils.data import Subset
        local_ds = Subset(train_dataset, indices)

        # Watermarked clients: poison the local dataset
        if idx in cfg.WATERMARK_CLIENTS:
            local_ds = watermark.poison_dataset(local_ds, seed=idx)

        loader = DataLoader(local_ds, batch_size=cfg.LOCAL_BATCH_SIZE,
                            shuffle=True, drop_last=False)
        clients.append(
            Client(client_id=idx, dataloader=loader,
                   device=cfg.DEVICE,
                   local_epochs=cfg.LOCAL_EPOCHS,
                   lr=cfg.LOCAL_LR)
        )
    return clients


def make_eval_fn(clean_loader, trigger_loader, cfg):
    """Return an evaluation closure compatible with Server.train()."""
    def eval_fn(model, round_idx):
        return evaluate_all(
            model,
            clean_loader,
            trigger_loader,
            target_label=cfg.WATERMARK_TARGET_LABEL,
            device=cfg.DEVICE,
        )
    return eval_fn


def print_header(cfg, args):
    print("=" * 60)
    print("  Federated Model Watermarking Framework")
    print("=" * 60)
    print(f"  Dataset          : {cfg.DATASET}")
    print(f"  Model            : {cfg.MODEL}")
    print(f"  Clients          : {cfg.NUM_CLIENTS}  "
          f"(fraction per round: {cfg.CLIENT_FRACTION})")
    print(f"  Rounds           : {cfg.NUM_ROUNDS}")
    print(f"  Non-IID          : {cfg.NON_IID}  "
          f"(α={cfg.DIRICHLET_ALPHA})")
    print(f"  Watermark        : {cfg.WATERMARK_STRATEGY}")
    print(f"  Aggregation      : {cfg.AGGREGATION}")
    print("=" * 60)


def save_results(history, prune_results, cfg):
    """Persist experiment results as a simple text report."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(cfg.RESULTS_DIR, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Round,ACC,WSR,FTR\n")
        for i, m in enumerate(history, start=1):
            f.write(
                f"{i},"
                f"{m.get('acc', float('nan')):.4f},"
                f"{m.get('wsr', float('nan')):.4f},"
                f"{m.get('ftr', float('nan')):.4f}\n"
            )
        f.write("\nPruning Robustness\n")
        f.write("PruneRate,ACC,WSR,FTR\n")
        for m in prune_results:
            f.write(
                f"{m.get('prune_rate', 0):.2f},"
                f"{m.get('acc', float('nan')):.4f},"
                f"{m.get('wsr', float('nan')):.4f},"
                f"{m.get('ftr', float('nan')):.4f}\n"
            )
    print(f"\n  Results saved to: {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Federated Model Watermarking Framework"
    )
    parser.add_argument("--dataset", type=str,
                        help="Dataset name (MNIST or CIFAR10)")
    parser.add_argument("--model", type=str,
                        help="Model architecture (SimpleCNN or ResNet18)")
    parser.add_argument("--clients", type=int,
                        help="Number of federated clients")
    parser.add_argument("--rounds", type=int,
                        help="Number of global rounds")
    parser.add_argument("--strategy", type=str,
                        help="Watermark strategy (badnets or pattern)")
    parser.add_argument("--aggregation", type=str,
                        help="Aggregation method (fedavg, median, trimmed_mean)")
    parser.add_argument("--non-iid", action="store_true", default=None,
                        help="Use non-IID data partitioning")
    parser.add_argument("--iid", dest="non_iid", action="store_false",
                        help="Use IID data partitioning")
    parser.add_argument("--device", type=str,
                        help="Device (cpu or cuda)")
    parser.add_argument("--no-prune", action="store_true",
                        help="Skip pruning robustness evaluation")
    return parser.parse_args()


def apply_args_to_config(args, cfg):
    """Override config values with command-line arguments where provided."""
    if args.dataset:
        cfg.DATASET = args.dataset
    if args.model:
        cfg.MODEL = args.model
    if args.clients:
        cfg.NUM_CLIENTS = args.clients
    if args.rounds:
        cfg.NUM_ROUNDS = args.rounds
    if args.strategy:
        cfg.WATERMARK_STRATEGY = args.strategy
    if args.aggregation:
        cfg.AGGREGATION = args.aggregation
    if args.non_iid is not None:
        cfg.NON_IID = args.non_iid
    if args.device:
        cfg.DEVICE = args.device


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    apply_args_to_config(args, config)

    print_header(config, args)

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset …")
    train_ds, test_ds = load_dataset(config.DATASET, config.DATA_DIR)

    # ── 2. Partition data among clients ──────────────────────────────────────
    print("[2/5] Partitioning data among clients …")
    if config.NON_IID:
        client_indices = partition_non_iid_dirichlet(
            train_ds, config.NUM_CLIENTS,
            alpha=config.DIRICHLET_ALPHA,
        )
        print(f"  Non-IID (Dirichlet α={config.DIRICHLET_ALPHA}): "
              f"client sizes = {[len(idx) for idx in client_indices]}")
    else:
        client_indices = partition_iid(train_ds, config.NUM_CLIENTS)
        print(f"  IID: client sizes = {[len(idx) for idx in client_indices]}")

    # ── 3. Build watermark & clients ─────────────────────────────────────────
    print("[3/5] Building watermark and clients …")
    watermark = build_watermark(config.WATERMARK_STRATEGY, config)
    print(f"  Watermark: {watermark}")

    clients = build_clients(train_ds, client_indices, watermark, config)

    # Build evaluation data loaders
    clean_loader = get_test_dataloader(test_ds, config.EVAL_BATCH_SIZE)
    trigger_test_ds = watermark.build_trigger_test_dataset(test_ds)
    trigger_loader = DataLoader(trigger_test_ds,
                                batch_size=config.EVAL_BATCH_SIZE,
                                shuffle=False)

    # ── 4. Federated training ─────────────────────────────────────────────────
    print("[4/5] Starting federated training …")
    global_model = build_model(config.MODEL, config.DATASET, config.NUM_CLASSES)
    # Initialise lazy layers by running a dummy forward pass
    _dummy = torch.zeros(2, *train_ds[0][0].shape)
    global_model(_dummy)

    server = Server(
        global_model=global_model,
        clients=clients,
        aggregation=config.AGGREGATION,
        client_fraction=config.CLIENT_FRACTION,
        device=config.DEVICE,
    )
    eval_fn = make_eval_fn(clean_loader, trigger_loader, config)
    history = server.train(config.NUM_ROUNDS, eval_fn=eval_fn, verbose=True)

    final = history[-1] if history else {}
    print(
        f"\n  Final metrics — "
        f"ACC={final.get('acc', 0):.4f}  "
        f"WSR={final.get('wsr', 0):.4f}  "
        f"FTR={final.get('ftr', 0):.4f}"
    )

    # ── 5. Pruning robustness evaluation ────────────────────────────────────
    prune_results = []
    if not args.no_prune:
        print("\n[5/5] Pruning robustness evaluation …")
        final_model = server.get_global_model()

        def prune_eval(pruned_model):
            return evaluate_all(
                pruned_model, clean_loader, trigger_loader,
                target_label=config.WATERMARK_TARGET_LABEL,
                device=config.DEVICE,
            )

        prune_results = evaluate_pruning_robustness(
            final_model, config.PRUNE_RATES,
            eval_fn=prune_eval, verbose=True,
        )
    else:
        print("\n[5/5] Pruning evaluation skipped (--no-prune).")

    # ── Save ──────────────────────────────────────────────────────────────────
    if config.SAVE_MODEL:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        torch.save(server.get_global_model().state_dict(), config.MODEL_PATH)
        print(f"  Model saved to: {config.MODEL_PATH}")

    save_results(history, prune_results, config)
    print("\nDone.")


if __name__ == "__main__":
    main()
