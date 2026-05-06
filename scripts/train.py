"""
Train models from config files or CLI args.

Usage:
    python -m scripts.train --config configs/baseline.json
    python -m scripts.train --config configs/sweep_backbones.json
    python -m scripts.train --config configs/sweep_cnn_depth.json
    python -m scripts.train --model cnn --epochs 10             # quick single run
    python -m scripts.train --from-npz outputs/augmented_data.npz --config configs/baseline.json
"""

import argparse
import json
import multiprocessing
import os
import time

os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
multiprocessing.set_start_method('spawn', force=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data_loader import prepare_datasets, balance_classes, augment_data, IMG_CLASSES
from src.models import build_cnn_model, build_transfer_model


def train_model(model, x_train, y_train, x_valid, y_valid, epochs, batch_size, name):
    print(f'\n{"=" * 60}')
    print(f'Training {name} for {epochs} epochs (batch_size={batch_size})...')
    print(f'{"=" * 60}')

    start = time.time()
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        validation_data=(x_valid, y_valid),
        epochs=epochs,
        verbose=1,
        shuffle=True
    )
    elapsed = time.time() - start
    print(f'{name} training took {elapsed:.1f}s')
    return history, elapsed


def evaluate_model(model, x, y_onehot, label):
    score = model.evaluate(x, y_onehot, verbose=0)
    print(f'{label} -> Loss: {score[0]:.4f}  Accuracy: {score[1] * 100:.2f}%')
    return score


def save_plots(histories, output_dir, experiment_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(experiment_name, fontsize=14)

    for name, h in histories.items():
        ax1.plot(h.history['accuracy'], label=f'{name} train')
        ax1.plot(h.history['val_accuracy'], label=f'{name} val', linestyle='--')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(fontsize=8)
    ax1.grid(True)

    for name, h in histories.items():
        ax2.plot(h.history['loss'], label=f'{name} train')
        ax2.plot(h.history['val_loss'], label=f'{name} val', linestyle='--')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=8)
    ax2.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f'Saved training curves to {plot_path}')


def build_model_from_config(name, model_cfg):
    model_type = model_cfg.pop('type', 'cnn')
    if model_type == 'transfer':
        return build_transfer_model(model_cfg)
    else:
        return build_cnn_model(model_cfg)


def load_data(args):
    if args.from_npz:
        print(f'Loading pre-augmented data from {args.from_npz}...')
        data = np.load(args.from_npz)
        return (data['x_train_aug'], data['y_train_aug'],
                data['x_valid'], data['y_valid'],
                data['x_test'], data['y_test'], data['y_test_onehot'])

    print('Loading and preparing dataset...')
    x_train, y_train, x_valid, y_valid, x_test, y_test, y_test_onehot = prepare_datasets(
        args.data, random_state=args.seed
    )

    if not args.no_balance:
        print('\nBalancing classes...')
        x_train, y_train = balance_classes(x_train, y_train, seed=args.seed)

    print('\nAugmenting data...')
    x_train_aug, y_train_aug, _ = augment_data(
        x_train, y_train, multiplier=args.multiplier, seed=args.seed,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False
    )
    return x_train_aug, y_train_aug, x_valid, y_valid, x_test, y_test, y_test_onehot


def run_experiment(config, x_train, y_train, x_valid, y_valid, x_test, y_test_onehot, output_dir):
    """Run all models defined in a config dict."""
    experiment_name = config.get('name', 'experiment')
    epochs = config.get('epochs', 20)
    batch_size = config.get('batch_size', 32)
    model_defs = config.get('models', {})

    exp_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    histories = {}
    results = {}

    for model_name, model_cfg in model_defs.items():
        cfg_copy = {**model_cfg}
        model = build_model_from_config(model_name, cfg_copy)

        history, elapsed = train_model(
            model, x_train, y_train, x_valid, y_valid,
            epochs, batch_size, model_name
        )
        histories[model_name] = history

        print(f'\n--- {model_name} Evaluation ---')
        train_score = evaluate_model(model, x_train, y_train, 'Train')
        val_score = evaluate_model(model, x_valid, y_valid, 'Valid')
        test_score = evaluate_model(model, x_test, y_test_onehot, 'Test ')

        results[model_name] = {
            'config': model_cfg,
            'train_acc': float(train_score[1]), 'train_loss': float(train_score[0]),
            'val_acc': float(val_score[1]), 'val_loss': float(val_score[0]),
            'test_acc': float(test_score[1]), 'test_loss': float(test_score[0]),
            'training_time_s': round(elapsed, 1),
        }

        model_path = os.path.join(exp_dir, f'{model_name}.keras')
        model.save(model_path)
        print(f'Saved {model_name} to {model_path}')

    # Summary table
    print(f'\n{"=" * 70}')
    print(f'Results: {experiment_name}')
    print(f'{"=" * 70}')
    print(f'{"Model":<20s} {"Train":>8s} {"Val":>8s} {"Test":>8s} {"Time":>8s}')
    print(f'{"-" * 20} {"-" * 8} {"-" * 8} {"-" * 8} {"-" * 8}')
    for name, r in results.items():
        print(f'{name:<20s} {r["train_acc"]*100:>7.2f}% {r["val_acc"]*100:>7.2f}% '
              f'{r["test_acc"]*100:>7.2f}% {r["training_time_s"]:>6.1f}s')

    if histories:
        save_plots(histories, exp_dir, experiment_name)

    results_path = os.path.join(exp_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved results to {results_path}')
    return results


def main():
    parser = argparse.ArgumentParser(description="Train CNN models for chest X-ray classification")
    parser.add_argument("--config", default=None, help="Path to experiment config JSON")
    parser.add_argument("--data", default="./data", help="Dataset root (default: ./data)")
    parser.add_argument("--from-npz", default=None, help="Load pre-augmented data from .npz")
    parser.add_argument("--model", choices=["cnn", "transfer", "both"], default="both",
                        help="Which model(s) to train when no config given (default: both)")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--multiplier", type=int, default=2, help="Augmentation multiplier (default: 2)")
    parser.add_argument("--no-balance", action="store_true", help="Skip minority class oversampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", default="outputs", help="Output directory (default: outputs)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data once
    x_train, y_train, x_valid, y_valid, x_test, y_test, y_test_onehot = load_data(args)

    print(f'\nFinal training set: {x_train.shape}')
    print(f'Validation set:     {x_valid.shape}')
    print(f'Test set:           {x_test.shape}')

    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        # CLI overrides
        if '--epochs' in ' '.join(os.sys.argv):
            config['epochs'] = args.epochs
        if '--batch-size' in ' '.join(os.sys.argv):
            config['batch_size'] = args.batch_size
        run_experiment(config, x_train, y_train, x_valid, y_valid,
                       x_test, y_test_onehot, args.output_dir)
    else:
        # Fallback: build default config from CLI args
        config = {
            'name': 'cli_run',
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'models': {}
        }
        if args.model in ('cnn', 'both'):
            config['models']['cnn'] = {'type': 'cnn'}
        if args.model in ('transfer', 'both'):
            config['models']['transfer'] = {'type': 'transfer', 'backbone': 'VGG16'}
        run_experiment(config, x_train, y_train, x_valid, y_valid,
                       x_test, y_test_onehot, args.output_dir)


if __name__ == "__main__":
    main()
