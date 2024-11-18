from enum import Enum

import tensorflow as tf


class ICHFeature(Enum):
    # Image
    # # Baseline
    BASELINE_IMAGE = ('baseline/image/data', tf.string)
    BASELINE_IMAGE_FILENAME = ('baseline/image/filename', tf.string)
    BASELINE_IMAGE_WIDTH = ('baseline/image/width', tf.int64)
    BASELINE_IMAGE_HEIGHT = ('baseline/image/height', tf.int64)
    BASELINE_IMAGE_SLICES_NUMBER = ('baseline/image/slices', tf.int64)
    BASELINE_RESIZE_IMAGE = ('baseline/image/resize/data', tf.string)
    BASELINE_RESIZE_IMAGE_WIDTH = ('baseline/image/resize/width', tf.int64)
    BASELINE_RESIZE_IMAGE_HEIGHT = ('baseline/image/resize/height', tf.int64)
    BASELINE_RESIZE_IMAGE_SLICES_NUMBER = (
        'baseline/image/resize/slices', tf.int64)

    # # Followup
    FOLLOWUP_IMAGE = ('followup/image/data', tf.string)
    FOLLOWUP_IMAGE_FILENAME = ('followup/image/filename', tf.string)
    FOLLOWUP_IMAGE_WIDTH = ('followup/image/width', tf.int64)
    FOLLOWUP_IMAGE_HEIGHT = ('followup/image/height', tf.int64)
    FOLLOWUP_IMAGE_SLICES_NUMBER = ('followup/image/slices', tf.int64)
    FOLLOWUP_RESIZE_IMAGE = ('followup/image/resize/data', tf.string)
    FOLLOWUP_RESIZE_IMAGE_WIDTH = ('followup/image/resize/width', tf.int64)
    FOLLOWUP_RESIZE_IMAGE_HEIGHT = ('followup/image/resize/height', tf.int64)
    FOLLOWUP_RESIZE_IMAGE_SLICES_NUMBER = (
        'followup/image/resize/slices', tf.int64)

    # Label
    # # Baseline
    BASELINE_LABEL = ('baseline/label/data', tf.string)
    BASELINE_LABEL_FILENAME = ('baseline/label/filename', tf.string)
    BASELINE_LABEL_SLICES_NUMBER = ('baseline/label/slices', tf.int64)
    BASELINE_RESIZE_LABEL = ('baseline/label/resize/data', tf.string)
    BASELINE_RESIZE_LABEL_SLICES_NUMBER = (
        'baseline/label/resize/slices', tf.int64)
    # # Followup
    FOLLOWUP_LABEL = ('followup/label/data', tf.string)
    FOLLOWUP_LABEL_FILENAME = ('followup/label/filename', tf.string)
    FOLLOWUP_LABEL_SLICES_NUMBER = ('followup/label/slices', tf.int64)
    FOLLOWUP_RESIZE_LABEL = ('followup/label/resize/data', tf.string)
    FOLLOWUP_RESIZE_LABEL_SLICES_NUMBER = (
        'followup/label/resize/slices', tf.int64)

    # Slope / Intercept
    BASELINE_SLOPE = ('baseline/slope', tf.float32)
    BASELINE_INTER = ('baseline/inter', tf.float32)
    FOLLOWUP_SLOPE = ('followup/slope', tf.float32)
    FOLLOWUP_INTER = ('followup/inter', tf.float32)

    # Hematoma Expansion
    HEMATOMA_EXPANSION = ('has/expansion', tf.int64)

    # Clinical Data
    CLINICAL_BASELINE_VOLUME = ('clinical/baseline/volume', tf.float32)
    CLINICAL_FOLLOWUP_VOLUME = ('clinical/followup/volume', tf.float32)
    CLINICAL_SEX = ('clinical/sex', tf.int64)
    CLINICAL_AGE = ('clinical/age', tf.float32)
    # # GCS
    CLINICAL_GCS = ('clinical/gcs', tf.float32)
    CLINICAL_GCS_EYE = ('clinical/gcs/eye', tf.int64)
    CLINICAL_GCS_VERBAL = ('clinical/gcs/verbal', tf.int64)
    CLINICAL_GCS_MOTOR = ('clinical/gcs/motor', tf.int64)
    # #
    CLINICAL_HYPERTENSION = ('clinical/hypertension', tf.int64)
    CLINICAL_DIABETES = ('clinical/diabetes', tf.int64)
    CLINICAL_UREMIA = ('clinical/uremia', tf.int64)
    CLINICAL_SMOKING = ('clinical/smoking', tf.int64)
    CLINICAL_ALCOHOL = ('clinical/alcohol', tf.int64)
    CLINICAL_DYSLIPIDEMIA = ('clinical/dyslipidemia', tf.int64)
    # # mRS_
    CLINICAL_MODIFY_RANK_SCALE = ('clinical/mrs', tf.int64)
    # # Location
    CLINICAL_LOCATION = ('clinical/location', tf.int64)

    # Master-Thesis
    MODEL_X_IMAGE_HEIGHT = ('model/x/image/height', tf.int64)
    MODEL_X_IMAGE_WIDTH = ('model/x/image/width', tf.int64)
    MODEL_X_IMAGE_DEPTH = ('model/x/image/depth', tf.int64)
    MODEL_1_BASELINE_IMAGE = ('model/1/baseline/data', tf.string)
    MODEL_2_1_BASELINE_IMAGE = ('model/2-1/baseline/data', tf.string)
    MODEL_2_2_BASELINE_IMAGE = ('model/2-2/baseline/data', tf.string)
    MODEL_3_1_BASELINE_IMAGE = ('model/3-1/baseline/data', tf.string)
    MODEL_3_2_BASELINE_IMAGE = ('model/3-2/baseline/data', tf.string)
    MODEL_4_BASELINE_IMAGE = ('model/4/baseline/data', tf.string)
    MODEL_5_BASELINE_IMAGE = ('model/5/baseline/data', tf.string)
