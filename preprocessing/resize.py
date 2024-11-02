import cv2
import numpy as np

from .preprocessor import Preprocessor


class Resize(Preprocessor):
    def __init__(self, size: tuple, dtype=np.uint8):
        super().__init__(dtype=dtype)
        self._size = size

    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if len(image.shape) > 2:
            result = []
            for _slice in image.T:
                result.append(self._resize(_slice))
            return np.asarray(result).T

        else:
            return self._resize(image)

    def _resize(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, dsize=self._size)
