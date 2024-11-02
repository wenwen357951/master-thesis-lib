import numpy as np

from .preprocessor import Preprocessor


class HounsfieldConverter(Preprocessor):
    def __init__(self, window_width: int, window_level: int, bit: int = 12,
                 dtype=np.uint16):
        super().__init__(dtype)
        self._window_width = window_width
        self._window_level = window_level
        self._bit = bit

    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        max_value = 2 ** self._bit
        a = max_value / self._window_width * kwargs['slope']
        b = (max_value / 2) - (max_value * self._window_level / self._window_width) + a * kwargs['intercept']

        converted_image = a * image + b
        return np.clip(converted_image, 0, max_value - 1)
