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
    BASELINE_RESIZE_IMAGE_SLICES_NUMBER = ('baseline/image/resize/slices', tf.int64)

    # # Followup
    FOLLOWUP_IMAGE = ('followup/image/data', tf.string)
    FOLLOWUP_IMAGE_FILENAME = ('followup/image/filename', tf.string)
    FOLLOWUP_IMAGE_WIDTH = ('followup/image/width', tf.int64)
    FOLLOWUP_IMAGE_HEIGHT = ('followup/image/height', tf.int64)
    FOLLOWUP_IMAGE_SLICES_NUMBER = ('followup/image/slices', tf.int64)
    FOLLOWUP_RESIZE_IMAGE = ('followup/image/resize/data', tf.string)
    FOLLOWUP_RESIZE_IMAGE_WIDTH = ('followup/image/resize/width', tf.int64)
    FOLLOWUP_RESIZE_IMAGE_HEIGHT = ('followup/image/resize/height', tf.int64)
    FOLLOWUP_RESIZE_IMAGE_SLICES_NUMBER = ('followup/image/resize/slices', tf.int64)

    # Label
    # # Baseline
    BASELINE_LABEL = ('baseline/label/data', tf.string)
    BASELINE_LABEL_FILENAME = ('baseline/label/filename', tf.string)
    BASELINE_LABEL_SLICES_NUMBER = ('baseline/label/slices', tf.int64)
    BASELINE_RESIZE_LABEL = ('baseline/label/resize/data', tf.string)
    BASELINE_RESIZE_LABEL_SLICES_NUMBER = ('baseline/label/resize/slices', tf.int64)
    # # Followup
    FOLLOWUP_LABEL = ('followup/label/data', tf.string)
    FOLLOWUP_LABEL_FILENAME = ('followup/label/filename', tf.string)
    FOLLOWUP_LABEL_SLICES_NUMBER = ('followup/label/slices', tf.int64)
    FOLLOWUP_RESIZE_LABEL = ('followup/label/resize/data', tf.string)
    FOLLOWUP_RESIZE_LABEL_SLICES_NUMBER = ('followup/label/resize/slices', tf.int64)

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
    CLINICAL_AGE = ('clinical/age', tf.int64)
    CLINICAL_GCS_EYE = ('clinical/gcs/eye', tf.int64)
    CLINICAL_GCS_VERBAL = ('clinical/gcs/verbal', tf.int64)
    CLINICAL_GCS_MOTOR = ('clinical/gcs/motor', tf.int64)
    CLINICAL_MODIFY_RANK_SCALE = ('clinical/mrs', tf.int64)
    CLINICAL_LOCATION = ('clinical/location', tf.int64)
