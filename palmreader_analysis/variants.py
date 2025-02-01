from enum import Enum


class Paw(Enum):
    LEFT_FRONT = "lfpaw"
    RIGHT_FRONT = "rfpaw"
    LEFT_HIND = "lhpaw"
    RIGHT_HIND = "rhpaw"

    def old_name(self) -> str:
        """
        Legacy luminance measurements use a different paw naming scheme. This function provides those old names.
        """
        if self.value == "lfpaw":
            return "front_left"
        elif self.value == "rfpaw":
            return "front_right"
        elif self.value == "lhpaw":
            return "hind_left"
        elif self.value == "rhpaw":
            return "hind_right"


class LuminanceMeasure(Enum):
    LUMINANCE = "luminance"
    LUMINESCENCE = "luminescence"
    PRINT_SIZE = "print_size"

    def units(self) -> str:
        if self == LuminanceMeasure.LUMINANCE:
            return "pixel intensity/area"
        elif self == LuminanceMeasure.LUMINESCENCE:
            return "pixel intensity"
        elif self == LuminanceMeasure.PRINT_SIZE:
            return "pixel area"

    def feature_name(self) -> str:
        # the feature for luminance is called luminance_rework but the summary column is just called luminance
        if self == LuminanceMeasure.LUMINANCE:
            return "luminance_rework"
        else:
            return self.value


class RatioOrder(Enum):
    RIGHT_OVER_LEFT = "right_over_left"
    LEFT_OVER_RIGHT = "left_over_right"

    def displayname(self) -> str:
        if self == RatioOrder.RIGHT_OVER_LEFT:
            return "r/l"
        elif self == RatioOrder.LEFT_OVER_RIGHT:
            return "l/r"
        else:
            raise ValueError(f"Invalid direction: {self}")

    def divide(self, left, right):
        if self == RatioOrder.RIGHT_OVER_LEFT:
            return right / left
        elif self == RatioOrder.LEFT_OVER_RIGHT:
            return left / right
        else:
            raise ValueError(f"Invalid direction: {self}")
