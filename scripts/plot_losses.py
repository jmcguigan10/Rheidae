#!/usr/bin/env python
"""
Parse training logs that contain lines like:
  Epoch 5: train_loss=0.1234 val_loss=0.2345
and plot train/val loss vs epoch.
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

LINE_RE = re.compile(
    r"Epoch\s+(\d+):\s*train_loss=([0-9.+-eE]+)\s+val_loss=([0-9.+-eE]+)"
)


def parse_log(path: Path):
    epochs, train, val = [], [], []
    for line in path.read_text().splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        ep, tr, va = m.groups()
        epochs.append(int(ep))
        train.append(float(tr))
        val.append(float(va))
    if not epochs:
        raise ValueError(f"No epoch lines found in {path}")
    return epochs, train, val


def plot(epochs, train, val, out_path: Path):
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train, label="train")
    plt.plot(epochs, val, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Val Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to log file with epoch lines.")
    ap.add_argument(
        "--out",
        default=None,
        help="Output PNG path for the plot. Defaults to results/loss_curve-<timestamp>.png",
    )
    args = ap.parse_args()

    log_path = Path(args.log)
    epochs, train, val = parse_log(log_path)
    if args.out:
        out_path = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = Path("results") / f"loss_curve-{ts}.png"
    plot(epochs, train, val, out_path)


if __name__ == "__main__":
    main()
