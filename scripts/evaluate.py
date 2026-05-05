"""
Evaluate a saved model on the test set and generate Kaggle predictions.

Usage:
    python -m scripts.evaluate outputs/transfer_model.keras
    python -m scripts.evaluate outputs/cnn_model.keras --kaggle
    python -m scripts.evaluate outputs/transfer_model.keras --from-npz outputs/augmented_data.npz
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from src.data_loader import prepare_datasets, load_kaggle_test, IMG_CLASSES


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved model")
    parser.add_argument("model_path", help="Path to saved .keras model")
    parser.add_argument("--data", default="./data", help="Dataset root (default: ./data)")
    parser.add_argument("--from-npz", default=None, help="Load test data from pre-saved .npz")
    parser.add_argument("--kaggle", action="store_true", help="Also predict on unlabeled Kaggle test set")
    parser.add_argument("--output-dir", default="outputs", help="Directory for outputs (default: outputs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading model from {args.model_path}...')
    model = load_model(args.model_path)

    if args.from_npz:
        print(f'Loading test data from {args.from_npz}...')
        data = np.load(args.from_npz)
        x_test = data['x_test']
        y_test = data['y_test']
        y_test_onehot = data['y_test_onehot']
    else:
        print('Loading dataset to extract test split...')
        _, _, _, _, x_test, y_test, y_test_onehot = prepare_datasets(
            args.data, random_state=args.seed
        )

    model_name = os.path.splitext(os.path.basename(args.model_path))[0]

    # --- Test evaluation ---
    print(f'\n{"=" * 60}')
    print(f'Evaluating {model_name} on test set ({len(x_test)} images)')
    print(f'{"=" * 60}')

    test_score = model.evaluate(x_test, y_test_onehot, verbose=0)
    print(f'Test Accuracy: {test_score[1] * 100:.2f}%')
    print(f'Test Loss:     {test_score[0]:.4f}')

    predictions = model.predict(x_test, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)

    # --- Classification report ---
    print(f'\n{"-" * 40}')
    print('Classification Report')
    print(f'{"-" * 40}')
    print(classification_report(y_test, pred_labels, target_names=IMG_CLASSES))

    # --- Confusion matrix ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred_labels, display_labels=IMG_CLASSES, cmap='Blues', ax=ax
    )
    ax.set_title(f'Confusion Matrix — {model_name}')
    cm_path = os.path.join(args.output_dir, f'{model_name}_confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f'Saved confusion matrix to {cm_path}')

    # --- Kaggle predictions ---
    if args.kaggle:
        print(f'\n{"=" * 60}')
        print('Generating Kaggle test predictions...')
        print(f'{"=" * 60}')
        x_kaggle = load_kaggle_test(os.path.join(args.data, 'test'))
        kaggle_preds = model.predict(x_kaggle, verbose=0)
        kaggle_labels = np.argmax(kaggle_preds, axis=1)
        csv_path = os.path.join(args.output_dir, 'prediction_labels.csv')
        np.savetxt(csv_path, kaggle_labels, delimiter=",", fmt='%d')
        print(f'Saved {len(kaggle_labels)} predictions to {csv_path}')


if __name__ == "__main__":
    main()
