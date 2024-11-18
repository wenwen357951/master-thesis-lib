from .ich_he_case import HECase
from .glasgow_coma_scale import GlasgowComaScale
from ..enums import HELocation


class HEClinicalCase(HECase):

    def __init__(
            # Inherit
            self, case_name: str,
            baseline_image_filepath: str,
            followup_image_filepath: str,
            baseline_label_filepath: str,
            followup_label_filepath: str,
            baseline_he_pixel_number: int,
            followup_he_pixel_number: int,
            baseline_he_volume: float,
            followup_he_volume: float,
            baseline_followup_he_volume_difference: float,
            volume_difference_ratio: float,
            volume_difference_greater_6cc: bool,
            volume_difference_greater_33percent: bool,
            is_hematoma_expansion: bool,
            # HEClinicalCase
            sex: int,
            age: float,
            gcs: float,
            hypertension: int,
            diabetes: int,
            uremia: int,
            smoking: int,
            alcohol: int,
            dyslipidemia: int,
            he_location: int
    ):
        super().__init__(
            case_name,
            baseline_image_filepath,
            followup_image_filepath,
            baseline_label_filepath,
            followup_label_filepath,
            baseline_he_pixel_number,
            followup_he_pixel_number,
            baseline_he_volume,
            followup_he_volume,
            baseline_followup_he_volume_difference,
            volume_difference_ratio,
            volume_difference_greater_6cc,
            volume_difference_greater_33percent,
            is_hematoma_expansion
        )
        self._sex = sex
        self._age = age
        self._gcs = gcs
        self._hypertension = hypertension
        self._diabetes = diabetes
        self._uremia = uremia
        self._smoking = smoking
        self._alcohol = alcohol
        self._dyslipidemia = dyslipidemia
        self._he_location = he_location

    @property
    def sex(self) -> int:
        return self._sex

    @property
    def age(self) -> float:
        return self._age

    @property
    def gcs(self) -> float:
        return self._gcs

    @property
    def hypertension(self) -> int:
        return self._hypertension

    @property
    def diabetes(self) -> int:
        return self._diabetes

    @property
    def uremia(self) -> int:
        return self._uremia

    @property
    def smoking(self) -> int:
        return self._smoking

    @property
    def alcohol(self) -> int:
        return self._alcohol

    @property
    def dyslipidemia(self) -> int:
        return self._dyslipidemia

    @property
    def he_location(self) -> int:
        return self._he_location
