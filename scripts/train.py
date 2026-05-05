"""
Train baseline CNN and VGG16 transfer learning models.

Usage:
    python -m scripts.train                                     # full pipeline
    python -m scripts.train --model cnn                         # train only baseline
    python -m scripts.train --model transfer                    # train only VGG16
    python -m scripts.train --epochs 30 --batch-size 64         # custom hyperparams
    python -m scripts.train --from-npz outputs/augmented_data.npz  # use pre-augmented data
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data_loader import prepare_datasets, balance_classes, augment_data, load_kaggle_test, IMG_CLASSES
from src.models import build_cnn_model, build_transfer_model


def train_model(model, x_train, y_train, x_valid, y_valid, epochs, batch_size, name):
    print(f'\n{"=" * 60}')
    print(f'Training {name} for {epochs} epochs (batch_size={batch_size})...')
    print(f'{"=" * 60}')

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        validation_data=(x_valid, y_valid),
        epochs=epochs,
        verbose=1,
        shuffle=True
    )
    return history


def evaluate_model(model, x, y_onehot, label):
    score = model.evaluate(x, y_onehot, verbose=0)
    print(f'{label} -> Loss: {score[0]:.4f}  Accuracy: {score[1] * 100:.2f}%')
    return score


def save_plots(histories, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, h in histories.items():
        ax1.plot(h.history['accuracy'], label=f'{name} train')
        ax1.plot(h.history['val_accuracy'], label=f'{name} val', linestyle='--')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    for name, h in histories.items():
        ax2.plot(h.history['loss'], label=f'{name} train')
        ax2.plot(h.history['val_loss'], label=f'{name} val', linestyle='--')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f'Saved training curves to {plot_path}')


def main():
    parser = argparse.ArgumentParser(description="Train CNN models for chest X-ray classification")
    parser.add_argument("--data", default="./data", help="Dataset root (default: ./data)")
    parser.add_argument("--from-npz", default=None, help="Load pre-augmented data from .npz instead of processing from scratch")
    parser.add_argument("--model", choices=["cnn", "transfer", "both"], default="both", help="Which model(s) to train (default: both)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--multiplier", type=int, default=2, help="Augmentation multiplier (default: 2)")
    parser.add_argument("--no-balance", action="store_true", help="Skip minority class oversampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", default="outputs", help="Directory for saved models and plots (default: outputs)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.from_npz:
        print(f'Loading pre-augmented data from {args.from_npz}...')
        data = np.load(args.from_npz)
        x_train_aug = data['x_train_aug']
        y_train_aug = data['y_train_aug']
        x_valid = data['x_valid']
        y_valid = data['y_valid']
        x_test = data['x_test']
        y_test = data['y_test']
        y_test_onehot = data['y_test_onehot']
    else:
        print('Loading and preparing dataset...')
        x_train, y_train, x_valid, y_valid, x_test, y_test, y_test_onehot = prepare_datasets(
            args.data, random_state=args.seed
        )

        if not args.no_balance:
            print('\nBalancing classes...')
            x_train, y_train = balance_classes(x_train, y_train, seed=args.seed)

        print('\nAugmenting data...')
        x_train_aug, y_train_aug, _ = augment_data(
            x_train, y_train, multiplier=args.multiplier, seed=args.seed
        )

    print(f'\nFinal training set: {x_train_aug.shape}')
    print(f'Validation set:     {x_valid.shape}')
    print(f'Test set:           {x_test.shape}')

    histories = {}
    results = {}

    # --- Baseline CNN ---
    if args.model in ('cnn', 'both'):
        cnn_model = build_cnn_model()
        cnn_history = train_model(
            cnn_model, x_train_aug, y_train_aug, x_valid, y_valid,
            args.epochs, args.batch_size, 'Baseline CNN'
        )
        histories['CNN'] = cnn_history

        print('\n--- Baseline CNN Evaluation ---')
        train_score = evaluate_model(cnn_model, x_train_aug, y_train_aug, 'Train')
        val_score = evaluate_model(cnn_model, x_valid, y_valid, 'Valid')
        test_score = evaluate_model(cnn_model, x_test, y_test_onehot, 'Test ')

        results['cnn'] = {
            'train_acc': float(train_score[1]), 'train_loss': float(train_score[0]),
            'val_acc': float(val_score[1]), 'val_loss': float(val_score[0]),
            'test_acc': float(test_score[1]), 'test_loss': float(test_score[0]),
        }

        cnn_path = os.path.join(args.output_dir, 'cnn_model.keras')
        cnn_model.save(cnn_path)
        print(f'Saved baseline CNN to {cnn_path}')

    # --- VGG16 Transfer Learning ---
    if args.model in ('transfer', 'both'):
        transfer_model = build_transfer_model()
        transfer_history = train_model(
            transfer_model, x_train_aug, y_train_aug, x_valid, y_valid,
            args.epochs, args.batch_size, 'VGG16 Transfer'
        )
        histories['VGG16'] = transfer_history

        print('\n--- VGG16 Transfer Evaluation ---')
        train_score = evaluate_model(transfer_model, x_train_aug, y_train_aug, 'Train')
        val_score = evaluate_model(transfer_model, x_valid, y_valid, 'Valid')
        test_score = evaluate_model(transfer_model, x_test, y_test_onehot, 'Test ')

        results['transfer'] = {
            'train_acc': float(train_score[1]), 'train_loss': float(train_score[0]),
            'val_acc': float(val_score[1]), 'val_loss': float(val_score[0]),
            'test_acc': float(test_score[1]), 'test_loss': float(test_score[0]),
        }

        transfer_path = os.path.join(args.output_dir, 'transfer_model.keras')
        transfer_model.save(transfer_path)
        print(f'Saved VGG16 transfer model to {transfer_path}')

    # --- Comparison ---
    if len(histories) > 1:
        print(f'\n{"=" * 60}')
        print('Model Comparison (Test Set)')
        print(f'{"=" * 60}')
        for name, r in results.items():
            print(f'  {name:10s}  Acc: {r["test_acc"] * 100:.2f}%  Loss: {r["test_loss"]:.4f}')

    if histories:
        save_plots(histories, args.output_dir)

    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved results to {results_path}')


if __name__ == "__main__":
    main()
