"""
Notebook-friendly launcher for Group A training on Colab/Kaggle.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Group A with sane notebook defaults")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--sample-fraction", type=float, default=None)
    parser.add_argument("--max-devices", type=int, default=None)
    parser.add_argument("--residual-root", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Detected device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    project_root = Path(__file__).resolve().parents[1]
    cmd = [
        "python",
        str(project_root / "experiments" / "run_group_A.py"),
        "--config",
        args.config,
        "--device",
        device,
        "--num-workers",
        "4" if device == "cuda" else "2",
        "--prefetch-factor",
        "2",
    ]
    if args.sample_fraction is not None:
        cmd += ["--sample-fraction", str(args.sample_fraction)]
    if args.max_devices is not None:
        cmd += ["--max-devices", str(args.max_devices)]
    if args.residual_root:
        cmd += ["--residual-root", args.residual_root]

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
