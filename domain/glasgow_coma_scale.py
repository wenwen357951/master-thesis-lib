class GlasgowComaScale(object):
    def __init__(self, eye_opening: int, verbal_response: str, motor_response: int):
        self._e = int(eye_opening)
        self._v = str(verbal_response)
        self._m = int(motor_response)

    @property
    def e(self) -> int:
        return self._e

    @property
    def v(self) -> int:
        try:
            return int(self._v)
        except ValueError:
            return 1

    @property
    def m(self) -> int:
        return self._m
