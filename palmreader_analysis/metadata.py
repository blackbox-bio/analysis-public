from enum import Enum
from typing import TypedDict, Union, List, Literal
from .variants import Side, Direction, Paw


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
    category: ColumnCategory
    tags: List[TagType]
    displayname: str
    description: str
    hidden: bool
    legacy: bool

    @staticmethod
    def make_hidden(category: ColumnCategory) -> "ColumnMetadata":
        return ColumnMetadata(
            category=category,
            tags=[],
            displayname="",
            description="",
            hidden=True,
            legacy=False,
        )

    @staticmethod
    def make(
        category: ColumnCategory,
        tags: List[TagType],
        displayname: str,
        description: str,
        legacy: bool = False,
    ) -> "ColumnMetadata":
        return ColumnMetadata(
            category=category,
            tags=tags,
            displayname=displayname,
            description=description,
            hidden=False,
            legacy=False,
        )
