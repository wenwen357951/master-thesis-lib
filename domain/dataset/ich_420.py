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
