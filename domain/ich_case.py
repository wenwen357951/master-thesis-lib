from .nifti import NIFTIImage

class ICHCase(object):
    def __init__(
            self, case_name: str,
            baseline_image_filepath: str, followup_image_filepath: str,
            baseline_label_filepath: str, followup_label_filepath: str
    ):
        self._case_name = case_name
        self._baseline_image_filepath = baseline_image_filepath
        self._followup_image_filepath = followup_image_filepath
        self._baseline_label_filepath = baseline_label_filepath
        self._followup_label_filepath = followup_label_filepath

    @property
    def case_name(self):
        return self._case_name

    @property
    def baseline_image_filepath(self):
        return self._baseline_image_filepath

    @property
    def followup_image_filepath(self):
        return self._followup_image_filepath

    @property
    def baseline_label_filepath(self):
        return self._baseline_label_filepath

    @property
    def followup_label_filepath(self):
        return self._followup_label_filepath

    def baseline_image(self):
        return NIFTIImage(self._baseline_image_filepath)

    def followup_image(self):
        return NIFTIImage(self._followup_image_filepath)

    def baseline_label(self):
        return NIFTIImage(self._baseline_label_filepath)

    def followup_label(self):
        return NIFTIImage(self._followup_label_filepath)
