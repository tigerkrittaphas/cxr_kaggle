"""
Pre-generate augmented training data and save to disk as .npz files.

Usage:
    python -m scripts.augment                           # defaults
    python -m scripts.augment --multiplier 3            # 3x augmented copies
    python -m scripts.augment --output outputs/aug.npz  # custom output path

The saved .npz contains:
    x_train_aug  — augmented + original training images  (float32, 0-1)
    y_train_aug  — corresponding one-hot labels
    x_valid, y_valid, x_test, y_test, y_test_onehot — unchanged splits
"""

import argparse
import multiprocessing
import os

os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
multiprocessing.set_start_method('spawn', force=True)

import numpy as np

from src.data_loader import prepare_datasets, balance_classes, augment_data


def main():
    parser = argparse.ArgumentParser(description="Run data augmentation and save arrays to disk")
    parser.add_argument("--data", default="./data", help="Path to dataset root (default: ./data)")
    parser.add_argument("--multiplier", type=int, default=2, help="Augmented copies per original (default: 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--test-size", type=float, default=0.1, help="Test fraction (default: 0.1)")
    parser.add_argument("--valid-size", type=float, default=0.2, help="Validation fraction of train pool (default: 0.2)")
    parser.add_argument("--output", default="outputs/augmented_data.npz", help="Output .npz path (default: outputs/augmented_data.npz)")
    parser.add_argument("--no-balance", action="store_true", help="Skip minority class oversampling")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("=" * 60)
    print("Loading and splitting dataset...")
    print("=" * 60)
    x_train, y_train, x_valid, y_valid, x_test, y_test, y_test_onehot = prepare_datasets(
        args.data, test_size=args.test_size, valid_size=args.valid_size, random_state=args.seed
    )

    if not args.no_balance:
        print()
        print("=" * 60)
        print("Balancing classes (oversampling minority)...")
        print("=" * 60)
        x_train, y_train = balance_classes(x_train, y_train, seed=args.seed)

    print()
    print("=" * 60)
    print(f"Augmenting data (multiplier={args.multiplier}, seed={args.seed})...")
    print("=" * 60)
    x_train_aug, y_train_aug, _ = augment_data(
        x_train, y_train, multiplier=args.multiplier, seed=args.seed
    )

    print()
    print("=" * 60)
    print(f"Saving to {args.output}...")
    print("=" * 60)
    np.savez_compressed(
        args.output,
        x_train_aug=x_train_aug,
        y_train_aug=y_train_aug,
        x_valid=x_valid,
        y_valid=y_valid,
        x_test=x_test,
        y_test=y_test,
        y_test_onehot=y_test_onehot,
    )

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Saved {args.output} ({file_size_mb:.1f} MB)")
    print()
    print("Arrays in file:")
    print(f"  x_train_aug:   {x_train_aug.shape}  dtype={x_train_aug.dtype}")
    print(f"  y_train_aug:   {y_train_aug.shape}  dtype={y_train_aug.dtype}")
    print(f"  x_valid:       {x_valid.shape}  dtype={x_valid.dtype}")
    print(f"  y_valid:       {y_valid.shape}  dtype={y_valid.dtype}")
    print(f"  x_test:        {x_test.shape}  dtype={x_test.dtype}")
    print(f"  y_test:        {y_test.shape}  dtype={y_test.dtype}")
    print(f"  y_test_onehot: {y_test_onehot.shape}  dtype={y_test_onehot.dtype}")
    print()
    print("Load in notebook with:")
    print("  data = np.load('outputs/augmented_data.npz')")
    print("  x_train_aug = data['x_train_aug']")


if __name__ == "__main__":
    main()
