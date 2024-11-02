from enum import Enum

from keras import applications as KA


class Backbone(Enum):
    EfficientNetV2B3 = 'EfficientNetV2B3'
    ResNet101V2 = 'ResNet101V2'
    ResNet50V2 = 'ResNet50V2'


def get_backbone(backbone: Backbone, input_shape=(512, 512, 3), weights='imagenet'):
    if backbone == backbone.EfficientNetV2B3:
        return KA.EfficientNetV2B3(
            input_shape=input_shape,
            weights=weights,
            include_top=False
        )
    elif backbone == backbone.ResNet101V2:
        return KA.ResNet101V2(
            input_shape=input_shape,
            weights=weights,
            include_top=False
        )
    elif backbone == backbone.ResNet50V2:
        return KA.ResNet50V2(
            input_shape=input_shape,
            weights=weights,
            include_top=False
        )
