import os

import keras
import tensorflow as tf


class Config(object):
    RANDOM_SEED = 9487

    # Project
    ROOD_DIR = os.path.abspath(os.getcwd())
    RESOURCE_DIR = os.path.join(ROOD_DIR, 'resource')
    OUTPUT_DIR = os.path.join(ROOD_DIR, 'output')
    LOGS_DIR = os.path.join(ROOD_DIR, 'logs')

    # Dataset
    DATASET_DIR = os.path.abspath('/Volumes/SWChen-T7/datasets/ICH_420')
    DATASET_IMAGES_DIR = os.path.join(DATASET_DIR, 'Images')
    DATASET_LABELS_DIR = os.path.join(DATASET_DIR, 'Labels')
    DATASET_TFRECORD_DIR = os.path.join(DATASET_DIR, 'TFRecord')
    DATASET_CSV = os.path.join(RESOURCE_DIR, 'pair_he_clinical.csv')
    TRAIN_SET_RATIO = 0.8
    VALID_SET_RATIO = 0.1
    TEST_SET_RATIO = 0.1

    # Training
    BATCH_SIZE = 16
    EPOCH = 100
    INPUT_SHAPE = (512, 512, 28)
    INPUT_CHANNEL = 1
    BACKBONE = 'resnet18'
    MODEL_NAME = "MODEL_NAME"
    INCLUDE_CLINICAL_DATA = False
    MODEL_FCL = [256]
    DROPOUT = 0.3
    _MODEL_NAME = f"{MODEL_NAME}_{BACKBONE}"
    MODEL_OUTPUT_DIR = os.path.join(LOGS_DIR, _MODEL_NAME)

    # Tensorflow
    TF_AUTOTUNE = tf.data.AUTOTUNE

    # TFRecord
    TFRECORD_NAME = 'general'
    TFRECORD_SHARD = 4
    TFRECORD_OPTIONS = tf.io.TFRecordOptions(
        compression_type='GZIP',
        compression_level=7
    )

    def __init__(self):
        # Init
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.RESOURCE_DIR, exist_ok=True)

        self.DATASET_IMAGES_DIR = os.path.join(self.DATASET_DIR, 'Images')
        self.DATASET_LABELS_DIR = os.path.join(self.DATASET_DIR, 'Labels')
        self.DATASET_TFRECORD_DIR = os.path.join(self.DATASET_DIR, 'TFRecord')
        self._MODEL_NAME = f"{self.MODEL_NAME}_{self.BACKBONE}"
        self.MODEL_OUTPUT_DIR = os.path.join(
            self.LOGS_DIR,
            self._MODEL_NAME
        )
        print(f'Setting Random Seed: {self.RANDOM_SEED}')
        keras.utils.set_random_seed(self.RANDOM_SEED)

        # Display Information
        print(f"Model Name: {self.MODEL_NAME}")
        print(f"Backbone: {self.MODEL_NAME}")


if __name__ == '__main__':
    Config()
