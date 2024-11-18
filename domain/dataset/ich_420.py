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


def general_3D_clinical_feature_description():
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
        .append(ICHFeature.CLINICAL_BASELINE_VOLUME) \
        .append(ICHFeature.CLINICAL_FOLLOWUP_VOLUME) \
        .append(ICHFeature.CLINICAL_SEX) \
        .append(ICHFeature.CLINICAL_AGE) \
        .append(ICHFeature.CLINICAL_GCS) \
        .append(ICHFeature.CLINICAL_HYPERTENSION) \
        .append(ICHFeature.CLINICAL_DIABETES) \
        .append(ICHFeature.CLINICAL_UREMIA) \
        .append(ICHFeature.CLINICAL_SMOKING) \
        .append(ICHFeature.CLINICAL_ALCOHOL) \
        .append(ICHFeature.CLINICAL_DYSLIPIDEMIA) \
        .append(ICHFeature.CLINICAL_LOCATION) \
        .build()
