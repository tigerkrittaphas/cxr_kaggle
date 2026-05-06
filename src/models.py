from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Lambda
from keras import applications
import keras.backend as K

from src.data_loader import IMG_SIZE

BACKBONES = {
    'VGG16': applications.VGG16,
    'VGG19': applications.VGG19,
    'ResNet50': applications.ResNet50,
    'ResNet101': applications.ResNet101,
    'MobileNetV2': applications.MobileNetV2,
    'DenseNet121': applications.DenseNet121,
    'EfficientNetB0': applications.EfficientNetB0,
    'InceptionV3': applications.InceptionV3,
}

DEFAULT_CNN_CONFIG = {
    'conv_blocks': [
        {'filters': 32, 'kernel_size': 3, 'dropout': 0.0},
        {'filters': 64, 'kernel_size': 3, 'dropout': 0.25},
        {'filters': 128, 'kernel_size': 3, 'dropout': 0.25},
        {'filters': 256, 'kernel_size': 3, 'dropout': 0.25},
    ],
    'dense_layers': [128, 64],
    'dense_dropout': 0.5,
    'optimizer': 'adam',
    'learning_rate': None,
}

DEFAULT_TRANSFER_CONFIG = {
    'backbone': 'VGG16',
    'freeze_backbone': True,
    'dense_layers': [256, 128],
    'dense_dropout': 0.5,
    'optimizer': 'adam',
    'learning_rate': None,
}


def _get_optimizer(name, learning_rate=None):
    from keras import optimizers
    opts = {
        'adam': optimizers.Adam,
        'sgd': optimizers.SGD,
        'rmsprop': optimizers.RMSprop,
        'adamw': optimizers.AdamW,
    }
    cls = opts.get(name.lower(), optimizers.Adam)
    if learning_rate:
        return cls(learning_rate=learning_rate)
    return cls()


def build_cnn_model(config=None):
    cfg = {**DEFAULT_CNN_CONFIG, **(config or {})}

    model = Sequential()

    for i, block in enumerate(cfg['conv_blocks']):
        kwargs = dict(
            filters=block['filters'],
            kernel_size=block.get('kernel_size', 3),
            padding='same',
            activation='relu',
        )
        if i == 0:
            kwargs['input_shape'] = (IMG_SIZE, IMG_SIZE, 1)
        model.add(Conv2D(**kwargs))
        model.add(MaxPooling2D(pool_size=2))
        if block.get('dropout', 0) > 0:
            model.add(Dropout(block['dropout']))

    model.add(GlobalAveragePooling2D())

    for units in cfg['dense_layers']:
        model.add(Dense(units, activation='relu'))
        if cfg['dense_dropout'] > 0:
            model.add(Dropout(cfg['dense_dropout']))

    model.add(Dense(2, activation='softmax'))

    optimizer = _get_optimizer(cfg['optimizer'], cfg.get('learning_rate'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


def build_transfer_model(config=None):
    cfg = {**DEFAULT_TRANSFER_CONFIG, **(config or {})}

    backbone_name = cfg['backbone']
    if backbone_name not in BACKBONES:
        raise ValueError(f'Unknown backbone: {backbone_name}. Choose from: {list(BACKBONES.keys())}')

    backbone_cls = BACKBONES[backbone_name]
    pretrained = backbone_cls(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    if cfg['freeze_backbone']:
        for layer in pretrained.layers:
            layer.trainable = False

    model = Sequential()
    model.add(Lambda(lambda x: K.repeat_elements(x, 3, axis=-1),
                     input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(pretrained)
    model.add(GlobalAveragePooling2D())

    for units in cfg['dense_layers']:
        model.add(Dense(units, activation='relu'))
        if cfg['dense_dropout'] > 0:
            model.add(Dropout(cfg['dense_dropout']))

    model.add(Dense(2, activation='softmax'))

    optimizer = _get_optimizer(cfg['optimizer'], cfg.get('learning_rate'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
