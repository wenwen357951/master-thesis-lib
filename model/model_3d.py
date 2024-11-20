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


def get_3d_clinical_model(
        backbone_name: str = 'resnet34',
        weights: str = 'imagenet',
        input_shape: tuple = (512, 512, 25),
        input_channel: int = 3,
        full_connection_layer: list = [512, 256],
        class_num=1,
        with_hu_adjust=False,
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
    x = KL.Dense(512, activation='relu', kernel_initializer='he_normal')(x)
    x = KL.Dropout(dropout)(x)

    c = KL.Dense(512, kernel_initializer='he_normal')(input_clinical)
    c = KL.BatchNormalization()(c)
    c = KL.Activation("relu")(c)
    c = KL.Dropout(dropout)(c)

    c = KL.Dense(256, kernel_initializer='he_normal')(c)
    c = KL.BatchNormalization()(c)
    c = KL.Activation("relu")(c)
    c = KL.Dropout(dropout)(c)

    x = KL.concatenate([x, c])

    for index, node_num in enumerate(full_connection_layer):
        x = KL.Dense(
            node_num, kernel_initializer='he_normal', name=f'classification_{index}'
        )(x)
        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)
        x = KL.Dropout(dropout)(x)

    x = KL.Dense(class_num, activation='sigmoid', name='prediction')(x)

    model = keras.Model(
        inputs=(input_image, input_clinical),
        outputs=x
    )

    return model


def model_multimodal():

    inputs_image = keras.Input((512, 512, 28, 3))
    inputs_tabular = keras.Input((15,))

    t = KL.Dense(units=512, kernel_initializer="he_normal")(inputs_tabular)
    t = KL.BatchNormalization()(t)
    t = KL.Activation("relu")(t)
    t = KL.Dropout(0.3)(t)

    t = KL.Dense(units=256, kernel_initializer="he_normal")(t)
    t = KL.BatchNormalization()(t)
    t = KL.Activation("relu")(t)
    t = KL.Dropout(0.3)(t)

    i = KL.Conv3D(
        filters=64, kernel_size=(19, 19, 5), kernel_initializer="he_normal"
    )(inputs_image)
    i = KL.BatchNormalization()(i)
    i = KL.Activation("relu")(i)
    i = KL.MaxPool3D(pool_size=2)(i)

    i = KL.Conv3D(
        filters=128, kernel_size=(14, 14, 3), kernel_initializer="he_normal"
    )(i)
    i = KL.BatchNormalization()(i)
    i = KL.Activation("relu")(i)
    i = KL.MaxPool3D(pool_size=2)(i)

    i = KL.Conv3D(
        filters=256, kernel_size=(11, 11, 2), kernel_initializer="he_normal"
    )(i)
    i = KL.BatchNormalization()(i)
    i = KL.Activation("relu")(i)
    i = KL.MaxPool3D(pool_size=2)(i)

    i = KL.GlobalAveragePooling3D()(i)
    i = KL.Dense(
        units=512, activation="relu", kernel_initializer="he_normal"
    )(i)
    i = KL.Dropout(0.3)(i)

    concat = KL.concatenate([t, i])

    ti = KL.Dense(units=256, kernel_initializer="he_normal")(concat)
    ti = KL.BatchNormalization()(ti)
    ti = KL.Activation("relu")(ti)
    ti = KL.Dropout(0.3)(ti)

    outputs = KL.Dense(units=1, activation="sigmoid")(ti)

    return keras.Model(
        (inputs_image, inputs_tabular),
        outputs, name="multimodal"
    )
