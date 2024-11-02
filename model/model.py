import keras
from keras import applications as KA
from keras import layers as KL

from .backbone import get_backbone, Backbone
from .layers import WindowSetting


def get_resnet50v2_ap2d_pr_model(backbone: Backbone):
    backbone = get_backbone(backbone=backbone)

    # Input
    input_images = KL.Input(shape=(512, 512, 1), name='input_1')
    input_slopes = KL.Input(shape=(1,), name='input_2')
    input_inters = KL.Input(shape=(1,), name='input_3')

    adjust_hu = WindowSetting(90, 45)(input_images, slopes=input_slopes, inters=input_inters)
    adjust_hu = KA.resnet_v2.preprocess_input(adjust_hu)

    x = backbone(adjust_hu)
    x = KL.Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = KL.AveragePooling2D(pool_size=2, strides=2)(x)
    x = KL.Flatten()(x)
    x = KL.Dense(512, activation='relu')(x)
    x = KL.Dense(128, activation='relu')(x)
    x = KL.Dense(1, activation='sigmoid')(x)

    return keras.Model(inputs=[input_images, input_slopes, input_inters], outputs=x)


def get_resnet50v2_ap2d_pr_model_2(backbone: Backbone):
    backbone = get_backbone(backbone)

    # Input
    input_images = KL.Input(shape=(512, 512, 1), name='input_1')
    input_slopes = KL.Input(shape=(1,), name='input_2')
    input_inters = KL.Input(shape=(1,), name='input_3')

    adjust_hu = WindowSetting(90, 45)(input_images, slopes=input_slopes, inters=input_inters)
    adjust_hu = KA.resnet_v2.preprocess_input(adjust_hu)

    x = backbone(adjust_hu)
    x = KL.Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = KL.AveragePooling2D(pool_size=2, strides=2)(x)
    x = KL.Flatten()(x)
    x = KL.Dense(512, activation='relu')(x)
    x = KL.Dense(128, activation='relu')(x)
    x = KL.Dense(1, activation='sigmoid')(x)

    return keras.Model(inputs=[input_images, input_slopes, input_inters], outputs=x)
