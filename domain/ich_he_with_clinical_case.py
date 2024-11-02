from .ich_he_case import HECase
from .glasgow_coma_scale import GlasgowComaScale
from ..enums import HELocation


class HEClinicalCase(HECase):

    def __init__(
            # Inherit
            self, case_name: str,
            baseline_image_filepath: str, followup_image_filepath: str,
            baseline_label_filepath: str, followup_label_filepath: str,
            baseline_he_pixel_number: int, followup_he_pixel_number: int,
            baseline_he_volume: float, followup_he_volume: float,
            baseline_followup_he_volume_difference: float, volume_difference_ratio: float,
            volume_difference_greater_6cc: bool, volume_difference_greater_33percent: bool,
            is_hematoma_expansion: bool,
            # HEClinicalCase
            age: int,
            gcs: GlasgowComaScale,
            modify_rank_scale: int,
            he_location: HELocation
    ):
        super().__init__(case_name, baseline_image_filepath, followup_image_filepath, baseline_label_filepath,
                         followup_label_filepath
                         , baseline_he_pixel_number, followup_he_pixel_number,
                         baseline_he_volume, followup_he_volume, baseline_followup_he_volume_difference,
                         volume_difference_ratio, volume_difference_greater_6cc, volume_difference_greater_33percent,
                         is_hematoma_expansion)
        self._age = age
        self._gcs = gcs
        self._modify_rank_scale = modify_rank_scale
        self._he_location = he_location

    @property
    def age(self):
        return self._age

    @property
    def gcs(self):
        return self._gcs

    @property
    def modify_rank_scale(self):
        return self._modify_rank_scale

    @property
    def he_location(self):
        return self._he_location
