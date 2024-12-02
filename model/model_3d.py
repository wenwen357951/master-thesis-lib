from classification_models_3D.kkeras import Classifiers
from keras import layers as KL
from keras import regularizers as KR
import keras

from .layers import WindowSetting3D


def get_3d_model(
        backbone_name: str = 'resnet34',
        weights: str = 'imagenet',
        input_shape: tuple = (512, 512, 25),
        input_channel: int = 1,
        full_connection_layer: list = [512, 256],
        class_num=1,
        with_hu_adjust=False,
        dropout: float = 0.1
):
    backbone, preprocess_input = Classifiers.get(backbone_name)
    model = backbone(
        input_shape=input_shape + (3,),
        weights=weights,
        include_top=False
    )

    if with_hu_adjust:
        input_image = KL.Input(
            shape=input_shape + (input_channel,),
            name='input_0'
        )
        input_slope = KL.Input(shape=(1,), name='input_1')
        input_inter = KL.Input(shape=(1,), name='input_2')
        adjust_hu = WindowSetting3D(90, 45)(
            input_image,
            slopes=input_slope,
            inters=input_inter
        )
        adjust_hu = preprocess_input(adjust_hu)
        x = model(adjust_hu)
    else:
        input_image = KL.Input(
            shape=input_shape + (3,),
            name='input_0'
        )
        x = preprocess_input(input_image)
        x = model(x)

    x = KL.GlobalAveragePooling3D()(x)
    x = KL.Dropout(dropout)(x)

    for index, node_num in enumerate(full_connection_layer):
        x = KL.Dense(
            node_num,
            activation='leaky_relu',
            name=f'classification_{index}'
        )(x)

    x = KL.Dense(class_num, activation='sigmoid', name='prediction')(x)

    if with_hu_adjust:
        model = keras.Model(
            inputs=[
                input_image,
                input_slope,
                input_inter
            ], outputs=x
        )
    else:
        model = keras.Model(
            inputs=input_image,
            outputs=x
        )

    return model


def dense_block(units, dropout=0.3, initializer='he_normal', kernel_regularizer=None, name=None):
    def wrap(x):
        x = KL.Dense(
            units,
            kernel_initializer=initializer,
            kernel_regularizer=kernel_regularizer,
            name=name
        )(x)
        x = KL.BatchNormalization()(x)
        x = KL.Activation("relu")(x)
        x = KL.Dropout(dropout)(x)
        return x
    return wrap


def get_3d_clinical_model(
        backbone_name: str = 'resnet34',
        weights: str = 'imagenet',
        input_shape: tuple = (512, 512, 28),
        input_channel: int = 1,
        full_connection_layer: list = [256],
        class_num=1,
        dropout: float = 0.1
):
    print('input_shape', input_shape + (input_channel,))
    backbone, preprocess_input = Classifiers.get(backbone_name)
    model = backbone(
        input_shape=input_shape + (input_channel,),
        weights=weights,
        include_top=False
    )
    input_image = KL.Input(
        shape=input_shape + (input_channel,),
        name='input_image'
    )
    input_clinical = KL.Input(
        shape=(15,),
        name='input_clinical'
    )

    x = preprocess_input(input_image)
    x = model(x)
    x = KL.GlobalAveragePooling3D()(x)
    x = KL.Dense(
        512,
        activation='relu',
        kernel_initializer='he_normal'
    )(x)
    x = KL.Dropout(dropout)(x)

    c = dense_block(
        256,
        dropout=dropout,
        kernel_regularizer=KR.L1L2(1e-3, 1e-3)
    )(input_clinical)
    c = dense_block(
        128,
        dropout=dropout,
        kernel_regularizer=KR.L1L2(1e-3, 1e-3)
    )(c)
    x = KL.concatenate([x, c])

    for index, units in enumerate(full_connection_layer):
        x = dense_block(
            units,
            dropout=dropout,
            kernel_regularizer=KR.L1L2(1e-3, 1e-3),
            name=f'classification_{index}'
        )(x)

    x = KL.Dense(class_num, activation='sigmoid', name='prediction')(x)
    model = keras.Model(
        inputs=(input_image, input_clinical),
        outputs=x
    )
    return model
