import logging

import tensorflow as tf

from .utils import _bytes_feature
from .utils import _float_feature
from .utils import _int64_feature
from ..enums import ICHFeature


class FeatureBuilder(object):
    def __init__(self):
        self._features = dict()

    def append(self, ich_feature: ICHFeature, value):
        if ich_feature in self._features.keys():
            logging.error(f"Append Failed: The feature {ich_feature.name} already exists!")
            return self

        value_type = type(value)

        if ich_feature.value[1] in [tf.string, str, bytes]:
            return self._append_bytes(ich_feature, value)
        elif ich_feature.value[1] in [tf.bool, tf.int32, tf.uint32, tf.int64, tf.uint64, bool, int]:
            return self._append_int64(ich_feature, value)
        elif ich_feature.value[1] in [tf.float32, tf.float64, float]:
            return self._append_float(ich_feature, value)

        logging.warning(f"Append Failed: This value type '{value_type}' not supported!")
        return self

    def _append_int64(self, ich_feature: ICHFeature, value):
        self._features[ich_feature] = _int64_feature(value)
        logging.info(f"The feature '{ich_feature.name}' has been append!")
        return self

    def _append_bytes(self, ich_feature: ICHFeature, value):
        self._features[ich_feature] = _bytes_feature(value)
        logging.info(f"The feature '{ich_feature.name}' has been append!")
        return self

    def _append_float(self, ich_feature: ICHFeature, value):
        self._features[ich_feature] = _float_feature(value)
        logging.info(f"The feature '{ich_feature.name}' has been append!")
        return self

    def build(self):
        generate_result = dict()
        for key, val in self._features.items():
            generate_result[key.value[0]] = val

        return tf.train.Features(feature=generate_result)

    def __str__(self):
        return self._features.__str__()
