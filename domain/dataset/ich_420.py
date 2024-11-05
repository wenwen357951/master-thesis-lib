from ...enums import ICHFeature
from ...tfrecord import FeatureDescriptionBuilder


def general_feature_description():
    return FeatureDescriptionBuilder() \
        .append(ICHFeature.BASELINE_IMAGE) \
        .append(ICHFeature.BASELINE_LABEL) \
        .append(ICHFeature.BASELINE_SLOPE) \
        .append(ICHFeature.BASELINE_INTER) \
        .append(ICHFeature.HEMATOMA_EXPANSION) \
        .append(ICHFeature.BASELINE_IMAGE_WIDTH) \
        .append(ICHFeature.BASELINE_IMAGE_HEIGHT) \
        .append(ICHFeature.BASELINE_IMAGE_SLICES_NUMBER) \
        .append(ICHFeature.BASELINE_LABEL_SLICES_NUMBER) \
        .append(ICHFeature.BASELINE_RESIZE_IMAGE) \
        .append(ICHFeature.BASELINE_RESIZE_IMAGE_WIDTH) \
        .append(ICHFeature.BASELINE_RESIZE_IMAGE_HEIGHT) \
        .append(ICHFeature.BASELINE_RESIZE_IMAGE_SLICES_NUMBER) \
        .append(ICHFeature.BASELINE_RESIZE_LABEL) \
        .append(ICHFeature.BASELINE_RESIZE_LABEL_SLICES_NUMBER) \
        .build()

def general_3D_feature_description():
    return FeatureDescriptionBuilder() \
        .append(ICHFeature.BASELINE_IMAGE) \
        .append(ICHFeature.BASELINE_IMAGE_HEIGHT) \
        .append(ICHFeature.BASELINE_IMAGE_WIDTH) \
        .append(ICHFeature.BASELINE_IMAGE_SLICES_NUMBER) \
        .append(ICHFeature.BASELINE_IMAGE_FILENAME) \
        .append(ICHFeature.BASELINE_SLOPE) \
        .append(ICHFeature.BASELINE_INTER) \
        .append(ICHFeature.HEMATOMA_EXPANSION) \
        .append(ICHFeature.MODEL_1_BASELINE_IMAGE) \
        .append(ICHFeature.MODEL_2_1_BASELINE_IMAGE) \
        .append(ICHFeature.MODEL_2_2_BASELINE_IMAGE) \
        .append(ICHFeature.MODEL_4_BASELINE_IMAGE) \
        .append(ICHFeature.MODEL_X_IMAGE_HEIGHT) \
        .append(ICHFeature.MODEL_X_IMAGE_WIDTH) \
        .append(ICHFeature.MODEL_X_IMAGE_DEPTH) \
        .build()

