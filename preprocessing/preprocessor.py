from abc import ABC, abstractmethod

import numpy as np


class Preprocessor(ABC):
    def __init__(self, dtype=np.uint8):
        self._dtype = dtype

    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        return self._process(image, **kwargs).astype(self._dtype)

    @abstractmethod
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        pass
