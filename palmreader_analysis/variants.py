from enum import Enum
from typing import TypedDict, Union, List, Literal


class Side(Enum):
    LEFT = "left"
    RIGHT = "right"


class Direction(Enum):
    FORE = "fore"
    HIND = "hind"


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

    def side(self) -> Side:
        if self.value in ["lfpaw", "lhpaw"]:
            return Side.LEFT
        elif self.value in ["rfpaw", "rhpaw"]:
            return Side.RIGHT
        else:
            raise ValueError(f"Invalid paw: {self.value}")

    def direction(self) -> Direction:
        if self.value in ["lfpaw", "rfpaw"]:
            return Direction.FORE
        elif self.value in ["lhpaw", "rhpaw"]:
            return Direction.HIND
        else:
            raise ValueError(f"Invalid paw: {self.value}")

    def as_tag(self) -> "TagType":
        return {
            "kind": "position",
            "direction": self.direction().value,
            "side": self.side().value,
        }

    def displayname(self) -> str:
        side = self.side()
        direction = self.direction()

        return f"{side.value.capitalize()} {direction.value.capitalize()}"


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

    def displayname(self) -> str:
        if self == LuminanceMeasure.LUMINANCE:
            return "Pressure Index"
        elif self == LuminanceMeasure.LUMINESCENCE:
            return "Luminance"
        elif self == LuminanceMeasure.PRINT_SIZE:
            return "Print size"
        else:
            raise ValueError(f"Invalid measure: {self}")


class RatioOrder(Enum):
    RIGHT_OVER_LEFT = "right_over_left"
    LEFT_OVER_RIGHT = "left_over_right"

    def summary_display(self) -> str:
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

    def displayname(self) -> str:
        if self == RatioOrder.RIGHT_OVER_LEFT:
            return "Right over Left"
        elif self == RatioOrder.LEFT_OVER_RIGHT:
            return "Left over Right"
        else:
            raise ValueError(f"Invalid direction: {self}")


class ColumnCategory(Enum):
    LUMINANCE_BASED = "luminance_based"
    POSTURAL = "postural"
    TEMPORAL = "temporal"

    def displayname(self) -> str:
        if self == ColumnCategory.LUMINANCE_BASED:
            return "Luminance-based"
        elif self == ColumnCategory.POSTURAL:
            return "Postural"
        elif self == ColumnCategory.TEMPORAL:
            return "Temporal"

    def description(self) -> str:
        if self == ColumnCategory.LUMINANCE_BASED:
            return "Features that build upon paw luminance data"
        elif self == ColumnCategory.POSTURAL:
            return "Features that describe the relationship between various limbs"
        elif self == ColumnCategory.TEMPORAL:
            return "Features that measure the amount of time an animal spent in various positions"


class PositionTag(TypedDict):
    kind: Literal["position"]
    side: Side
    direction: Direction


TagType = Union[PositionTag]


def all_paws_tags() -> List[TagType]:
    return [
        {"kind": "position", "side": paw.side(), "direction": paw.direction()}
        for paw in Paw
    ]


class ColumnMetadata(TypedDict):
    column: str
    category: ColumnCategory
    tags: List[TagType]
    displayname: str
    description: str
    hidden: bool
    legacy: bool

    @staticmethod
    def make_hidden(column: str, category: ColumnCategory) -> "ColumnMetadata":
        return ColumnMetadata(
            column=column,
            category=category,
            tags=[],
            displayname="",
            description="",
            hidden=True,
            legacy=False,
        )

    @staticmethod
    def make(
        column: str,
        category: ColumnCategory,
        tags: List[TagType],
        displayname: str,
        description: str,
        legacy: bool = False,
    ) -> "ColumnMetadata":
        return ColumnMetadata(
            column=column,
            category=category,
            tags=tags,
            displayname=displayname,
            description=description,
            hidden=False,
            legacy=legacy,
        )
