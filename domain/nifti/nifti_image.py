import nibabel as nib
import numpy as np

from ...preprocessing.preprocessor import Preprocessor


class NIFTIImage(object):
    def __init__(self, nifti_filepath):
        self._nifti_filepath = nifti_filepath
        self._nifti_data: nib.nifti1.Nifti1Image = nib.load(nifti_filepath)
        self._preprocess_image = None

    @property
    def header(self) -> nib.nifti1.Nifti1Header:
        return self._nifti_data.header

    @property
    def slope(self):
        return self._nifti_data.dataobj.slope

    @property
    def intercept(self) -> float:
        return self._nifti_data.dataobj.inter

    def _get_unscaled(self) -> np.ndarray:
        result = self._nifti_data.dataobj.get_unscaled()
        result = np.swapaxes(result, 0, 1)[::-1, ::-1, :]
        return result

    def get_image(self, preprocessors: list[Preprocessor] = None) -> np.ndarray:
        self._preprocess_image = self._get_unscaled()

        if preprocessors is None:
            return self._preprocess_image

        for preprocessor in preprocessors:
            self._preprocess_image = preprocessor.process(
                self._preprocess_image,
                slope=self.slope,
                intercept=self.intercept
            )

        return self._preprocess_image
