"""
Quick script to train StyleGAN3 on Greek motifs.

Usage:
    # Basic training
    python scripts/train_gan.py

    # With custom config
    python scripts/train_gan.py --config configs/stylegan3_greek_simple.yaml

    # Resume from checkpoint
    python scripts/train_gan.py --resume models/checkpoints/checkpoint_epoch_0050.pt

    # Custom epochs/batch size
    python scripts/train_gan.py --epochs 100 --batch-size 4
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.train_stylegan3 import main

if __name__ == "__main__":
    main()

