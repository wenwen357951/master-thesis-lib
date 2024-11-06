import os

import glob2
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .config import Config
from .domain import GlasgowComaScale
from .domain import HECase
from .domain import HEClinicalCase
from .enums import HELocation


class ICH420Dataset(object):
    CSV_HEADER = [
        '名稱', 'Baseline影像檔案名稱', 'Follow-up影像檔案名稱', 'Baseline標記檔案名稱',
        'Follow-up標記檔案名稱', 'Baseline出血像數量', 'Baseline出血體積', 'Follow-up出血像數量',
        'Follow-up出血體積', 'Baseline與Follow-up出血差值', '體積變化>6cc', '體積變化率', '體積變化>33%',
        '是否發生血塊擴大', '臨床資料名稱', '性別', '年齡', 'GCS:E', 'GCS:V', 'GCS:M', '高血壓(Hypertension)',
        '糖尿病(Diabetes Mellitus)', '尿毒症(Uremia)', '吸菸(Smoking)', '酒精(Alcohol)',
        '血脂異常(Dyslipidemia)', '修改後雷氏量表(mRS)', 'GCS:V(E->1)',
        '出血位置(Location(基底核:1,丘腦:2,大腦皮質下:3,後顱窩:4))'
    ]

    def __init__(self, config: Config):
        self._config = config
        self._include_clinical = config.INCLUDE_CLINICAL_DATA
        self._csv_filepath = config.DATASET_CSV
        self._dataset_dir = config.DATASET_DIR
        # Init
        self._dataset_df = pd.read_csv(
            self._csv_filepath, encoding='utf-8', header=0, names=self.CSV_HEADER)
        self._image_dir = os.path.join(self._dataset_dir, 'Images')
        self._label_dir = os.path.join(self._dataset_dir, 'Labels')

        self._train_set, self._test_set = train_test_split(
            self.dataframe,
            test_size=config.TEST_SET_RATIO,
            random_state=Config.RANDOM_SEED,
            stratify=self.dataframe[["是否發生血塊擴大"]]
        )

    def _convert_object(self, item):
        case_name = str(item[0])
        baseline_image_filepath = os.path.join(
            self._config.DATASET_IMAGES_DIR, str(item[1]))
        followup_image_filepath = os.path.join(
            self._config.DATASET_IMAGES_DIR, str(item[2]))
        baseline_label_filepath = os.path.join(
            self._config.DATASET_LABELS_DIR, str(item[3]))
        followup_label_filepath = os.path.join(
            self._config.DATASET_LABELS_DIR, str(item[4]))

        if not self._include_clinical:
            return HECase(
                case_name,
                baseline_image_filepath,
                followup_image_filepath,
                baseline_label_filepath,
                followup_label_filepath,
                int(item[5]), int(item[7]),
                int(item[6]), int(item[8]),
                float(item[9]),
                float(item[11]),
                bool(item[10]),
                bool(item[12]),
                bool(item[13])
            )

        return HEClinicalCase(
            case_name,
            baseline_image_filepath,
            followup_image_filepath,
            baseline_label_filepath,
            followup_label_filepath,
            int(item[5]), int(item[7]),
            int(item[6]), int(item[8]),
            float(item[9]),
            float(item[11]),
            bool(item[10]),
            bool(item[12]),
            bool(item[13]),
            int(item[16]),
            GlasgowComaScale(
                item[17], item[18], item[19]
            ),
            int(item[26]),
            HELocation(int(item[28]))
        )

    @property
    def dataframe(self):
        if self._config.INCLUDE_CLINICAL_DATA:
            return self._dataset_df.dropna()

        return self._dataset_df

    @property
    def train_set_generator(self):
        for item in self._train_set.to_numpy():
            yield self._convert_object(item)

    @property
    def train_set_len(self):
        return len(self._train_set)

    @property
    def test_set_generator(self):
        for item in self._test_set.to_numpy():
            yield self._convert_object(item)

    @property
    def test_set_len(self):
        return len(self._test_set)

    def __len__(self):
        return len(self.dataframe)

    def __iter__(self):
        for item in self.dataframe.to_numpy():
            yield self._convert_object(item)

    def load_tfrecord(self, dataset_name: str, set_name, feature_description: dict):
        tfrecord_pattern = os.path.join(
            self._config.DATASET_TFRECORD_DIR,
            set_name,
            f'{dataset_name}_*_{set_name}_*.tfrecord'
        )
        print(f"TFRecord File Pattern: {tfrecord_pattern}")
        tfrecord_files = sorted(
            glob2.glob(tfrecord_pattern)
        )
        print(f"TFrecord Files: {tfrecord_files}")
        tfdataset = tf.data.TFRecordDataset(
            filenames=tfrecord_files,
            compression_type=self._config.TFRECORD_OPTIONS.compression_type,
            num_parallel_reads=self._config.TF_AUTOTUNE,
            name='Loading_TFRecord_Datasets'
        )

        def _decode_fn(example):
            return tf.io.parse_example(example, feature_description)

        return tfdataset.map(_decode_fn, num_parallel_calls=self._config.TF_AUTOTUNE)
