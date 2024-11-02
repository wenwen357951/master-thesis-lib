from .ich_case import ICHCase


class HECase(ICHCase):
    def __init__(
            # Inherit
            self, case_name: str,
            baseline_image_filepath: str, followup_image_filepath: str,
            baseline_label_filepath: str, followup_label_filepath: str,
            baseline_he_pixel_number: int, followup_he_pixel_number: int,
            # HECase
            baseline_he_volume: float, followup_he_volume: float,
            baseline_followup_he_volume_difference: float,
            volume_difference_ratio: float,
            volume_difference_greater_6cc: bool, volume_difference_greater_33percent: bool,
            is_hematoma_expansion: bool


    ):
        super().__init__(case_name, baseline_image_filepath, followup_image_filepath, baseline_label_filepath,
                         followup_label_filepath)
        self._baseline_he_pixel_number = baseline_he_pixel_number
        self._followup_he_pixel_number = followup_he_pixel_number
        self._baseline_he_volume = baseline_he_volume
        self._followup_he_volume = followup_he_volume
        self._baseline_followup_he_volume_difference = baseline_followup_he_volume_difference
        self._volume_difference_ratio = volume_difference_ratio
        self._volume_difference_greater_6cc = volume_difference_greater_6cc
        self._volume_difference_greater_33percent = volume_difference_greater_33percent
        self._is_hematoma_expansion = is_hematoma_expansion

    @property
    def baseline_he_pixel_number(self):
        return self._baseline_he_pixel_number

    @property
    def followup_he_pixel_number(self):
        return self._followup_he_pixel_number

    @property
    def baseline_he_volume(self):
        return self._baseline_he_volume

    @property
    def followup_he_volume(self):
        return self._followup_he_volume

    @property
    def baseline_followup_he_volume_difference(self):
        return self._baseline_followup_he_volume_difference

    @property
    def volume_difference_ratio(self):
        return self._volume_difference_ratio

    @property
    def volume_difference_greater_6cc(self):
        return self._volume_difference_greater_6cc

    @property
    def volume_difference_greater_33percent(self):
        return self._volume_difference_greater_33percent

    @property
    def is_hematoma_expansion(self):
        return self._is_hematoma_expansion
