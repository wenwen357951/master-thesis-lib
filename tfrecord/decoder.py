import tensorflow as tf

from ..enums import ICHFeature


def decode_image_3d(image: ICHFeature, height: ICHFeature, width: ICHFeature, depth: ICHFeature, dtype=tf.int16):
    def wrap(x: dict):
        raw_image = tf.io.decode_raw(
            x[image.value[0]], out_type=dtype
        )
        image_shape = (x[height.value[0]], x[width.value[0]], x[depth.value[0]])
        x[image.value[0]] = tf.reshape(raw_image, image_shape)
        return x

    return wrap
