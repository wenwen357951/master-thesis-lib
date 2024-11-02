import keras
import tensorflow as tf


class WindowSetting(keras.Layer):
    def __init__(self, ww, wl, bit=12):
        super().__init__()
        self.bit = bit
        self.windows_width = tf.constant(ww, dtype=tf.float32)
        self.windows_center = tf.constant(wl, dtype=tf.float32)
        self.__max_val = tf.constant(2 ** self.bit, dtype=tf.float32)

    def build(self, input_shape):
        pass

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, *args, **kwargs):
        images = tf.cast(inputs, dtype=tf.float32)
        intercepts = kwargs["inters"]
        slopes = kwargs["slopes"]

        _A = tf.divide(tf.multiply(slopes, self.__max_val), self.windows_width)
        _B = tf.add(tf.subtract(
            tf.divide(self.__max_val, 2),
            tf.divide(tf.multiply(self.__max_val, self.windows_center), self.windows_width)
        ), tf.multiply(_A, intercepts))

        result_image = tf.add(tf.multiply(images, _A[:, None, None]), _B[:, None, None])
        result_image = tf.clip_by_value(result_image, 0, tf.subtract(self.__max_val, 1))

        result_image = tf.squeeze(result_image, axis=-1)
        result_image = tf.stack([result_image, result_image, result_image], axis=-1)
        result_image = tf.cast(result_image, dtype=tf.float32) / self.__max_val

        return result_image

class WindowSetting3D(keras.Layer):
    def __init__(self, ww, wl, bit=12):
        super().__init__()
        self.bit = bit
        self.windows_width = tf.constant(ww, dtype=tf.float32)
        self.windows_center = tf.constant(wl, dtype=tf.float32)
        self.__max_val = tf.constant(2 ** self.bit, dtype=tf.float32)

    def build(self, input_shape):
        pass

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, *args, **kwargs):
        images = tf.cast(inputs, dtype=tf.float32)
        intercepts = kwargs["inters"]
        slopes = kwargs["slopes"]

        _A = tf.divide(tf.multiply(slopes, self.__max_val), self.windows_width)
        _B = tf.add(tf.subtract(
            tf.divide(self.__max_val, 2),
            tf.divide(tf.multiply(self.__max_val, self.windows_center), self.windows_width)
        ), tf.multiply(_A, intercepts))

        _A = tf.expand_dims(_A, axis=-1)
        _A = tf.expand_dims(_A, axis=-1)
        _A = tf.expand_dims(_A, axis=-1)

        _B = tf.expand_dims(_B, axis=-1)
        _B = tf.expand_dims(_B, axis=-1)
        _B = tf.expand_dims(_B, axis=-1)

        # (image * _A) + _B
        result_image = tf.multiply(images, _A)
        result_image = tf.add(result_image, _B)
        result_image = tf.clip_by_value(result_image, 0, tf.subtract(self.__max_val, 1))

        result_image = tf.squeeze(result_image, axis=-1)
        result_image = tf.stack([result_image, result_image, result_image], axis=-1)
        result_image = tf.cast(result_image, dtype=tf.float32) / self.__max_val

        return result_image
