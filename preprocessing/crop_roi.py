import cv2
import numpy as np

from .preprocessor import Preprocessor
from . import HounsfieldConverter


class CropROI(Preprocessor):
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        hu_converter = HounsfieldConverter(
            window_width=1,
            window_level=-300,
            bit=8,
            dtype=np.uint8
        )
        region_3d_image = hu_converter.process(image, **kwargs)
        _, region_threshold_3d_image = cv2.threshold(
            region_3d_image, 128, 255, cv2.THRESH_BINARY
        )

        roi_image = []
        for idx, slice_img in enumerate(region_threshold_3d_image.T):
            contours, _ = cv2.findContours(slice_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour_mask = np.zeros(slice_img.shape, dtype=np.uint8)
            if len(contours) > 0:
                max_contour_id = np.argmax([cv2.contourArea(contours[k]) for k in range(len(contours))])
                cv2.drawContours(
                    image=contour_mask,
                    contours=contours,
                    contourIdx=max_contour_id,
                    color=1,
                    thickness=cv2.FILLED
                )
                mask = cv2.dilate(contour_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))).astype(np.uint8)
                roi_image.append(image.T[idx] * mask)

            else:
                roi_image.append(slice_img)

        return np.asarray(roi_image).T
