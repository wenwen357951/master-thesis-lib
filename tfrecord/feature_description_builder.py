import tensorflow as tf

from ..enums import ICHFeature


class FeatureDescriptionBuilder(object):
    def __init__(self):
        self._feature_description = dict()

    def append(self, ich_feature: ICHFeature):
        self._feature_description[ich_feature] = tf.io.FixedLenFeature([], ich_feature.value[1])
        return self

    def build(self):
        _result = dict()
        for key, val in self._feature_description.items():
            _result[key.value[0]] = val

        return _result
