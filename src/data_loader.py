import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_CLASSES = ['NORMAL', 'PNEUMONIA']
IMG_SIZE = 150


def load_sample_imgs(path):
    imgs = []
    for cls in IMG_CLASSES:
        dir_path = os.path.join(path, cls)
        img_name = os.listdir(dir_path)[0]
        img = cv2.imread(os.path.join(dir_path, img_name))
        cls_index = IMG_CLASSES.index(cls)
        imgs.append([img, cls_index])
    return imgs


def load_data(path):
    data = []
    for cls in IMG_CLASSES:
        cls_path = os.path.join(path, cls)
        cls_index = IMG_CLASSES.index(cls)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append([img, cls_index])
    return data


def load_kaggle_test(path):
    data = []
    for img_name in sorted(os.listdir(path)):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
    features = np.array(data) / 255.0
    return features[..., np.newaxis]


def split_features_labels(data):
    features = np.array([item[0] for item in data]) / 255.0
    features = features[..., np.newaxis]  # (N, H, W) -> (N, H, W, 1)
    labels = np.array([item[1] for item in data])
    return features, labels


def _print_class_counts(y_onehot, label=""):
    counts = np.argmax(y_onehot, axis=1)
    for i, cls in enumerate(IMG_CLASSES):
        print(f'  {label}{cls}: {np.sum(counts == i)}')


def balance_classes(x_train, y_train, seed=42):
    """Oversample the minority class via augmentation until both classes are equal."""
    class_indices = np.argmax(y_train, axis=1)
    counts = [np.sum(class_indices == i) for i in range(len(IMG_CLASSES))]
    minority_idx = int(np.argmin(counts))
    majority_count = max(counts)
    minority_count = min(counts)
    deficit = majority_count - minority_count

    if deficit == 0:
        print('Classes already balanced.')
        return x_train, y_train

    print(f'Balancing: oversampling {IMG_CLASSES[minority_idx]} ({minority_count} -> {majority_count})')

    minority_mask = class_indices == minority_idx
    x_minority = x_train[minority_mask]
    y_minority = y_train[minority_mask]

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    np.random.seed(seed)
    aug_images = []
    aug_labels = []
    batch_size = min(32, len(x_minority))
    batches_needed = deficit // batch_size + 1
    gen = datagen.flow(x_minority, y_minority, batch_size=batch_size, seed=seed)

    for _ in range(batches_needed):
        imgs, labels = next(gen)
        aug_images.append(imgs)
        aug_labels.append(labels)

    aug_images = np.concatenate(aug_images)[:deficit]
    aug_labels = np.concatenate(aug_labels)[:deficit]

    x_balanced = np.concatenate([x_train, aug_images])
    y_balanced = np.concatenate([y_train, aug_labels])

    shuffle_idx = np.random.RandomState(seed).permutation(len(x_balanced))
    x_balanced = x_balanced[shuffle_idx]
    y_balanced = y_balanced[shuffle_idx]

    _print_class_counts(y_balanced, label='Balanced ')
    return x_balanced, y_balanced


def prepare_datasets(dataset_path, test_size=0.1, valid_size=0.2, random_state=42):
    """Load all labeled data, split into train/valid/test, and one-hot encode labels."""
    train_data = load_data(os.path.join(dataset_path, 'train'))
    valid_data = load_data(os.path.join(dataset_path, 'val'))
    all_data = train_data + valid_data

    train_pool, test_split = train_test_split(all_data, test_size=test_size, random_state=random_state)
    train_split, valid_split = train_test_split(train_pool, test_size=valid_size, random_state=random_state)

    x_train, y_train = split_features_labels(train_split)
    x_valid, y_valid = split_features_labels(valid_split)
    x_test, y_test = split_features_labels(test_split)

    y_train = to_categorical(y_train, num_classes=2)
    y_valid = to_categorical(y_valid, num_classes=2)
    y_test_onehot = to_categorical(y_test, num_classes=2)

    print(f'Training:   {len(x_train)} images  |  shape: {x_train.shape}')
    _print_class_counts(y_train, label='  ')
    print(f'Validation: {len(x_valid)} images  |  shape: {x_valid.shape}')
    print(f'Testing:    {len(x_test)} images  |  shape: {x_test.shape}')

    return x_train, y_train, x_valid, y_valid, x_test, y_test, y_test_onehot


def augment_data(x_train, y_train, multiplier=2, seed=42):
    """Pre-generate augmented data and combine with originals for reproducible training."""
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    np.random.seed(seed)
    aug_images = []
    aug_labels = []
    batch_size = 32

    batches_needed = (len(x_train) * multiplier) // batch_size + 1
    aug_gen = datagen.flow(x_train, y_train, batch_size=batch_size, seed=seed)

    for _ in range(batches_needed):
        batch_imgs, batch_labels = next(aug_gen)
        aug_images.append(batch_imgs)
        aug_labels.append(batch_labels)

    aug_images = np.concatenate(aug_images)[:len(x_train) * multiplier]
    aug_labels = np.concatenate(aug_labels)[:len(x_train) * multiplier]

    x_aug = np.concatenate([x_train, aug_images])
    y_aug = np.concatenate([y_train, aug_labels])

    shuffle_idx = np.random.RandomState(seed).permutation(len(x_aug))
    x_aug = x_aug[shuffle_idx]
    y_aug = y_aug[shuffle_idx]

    print(f'Original size: {len(x_train)}  ->  Augmented size: {len(x_aug)}')

    return x_aug, y_aug, datagen
