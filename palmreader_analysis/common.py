"""
Most features have a 1:1 relationship with some column in the summary CSV. In those cases, it makes the most sense to define the computation for both the feature and the summarized column in the same place. This module has all of those columns.

Any features which do not have a single column in the summary are defined in the `features.py` file, and likely have one or more summary columns defined in `summary.py`. This separation is not visible to the user. To get all columns for either features or summary, use `FeaturesContext.get_all_features` or `SummaryContext.get_all_columns`.
"""

from typing import List, Literal, Dict, Tuple, Union
import numpy as np
from .features import FeaturesContext, Feature
from .summary import SummaryColumn
from utils import cal_distance_, body_parts_distance, get_vector, get_angle


class DistanceDeltaDef(Feature, SummaryColumn):
    def extract(self, ctx: FeaturesContext):
        ctx._data["distance_delta"] = cal_distance_(ctx.label).reshape(-1)

    def summarize(self, ctx):
        ctx._data["distance_traveled (pixel)"] = np.nansum(
            ctx._features["distance_delta"]
        )


DISTANCE_FEATURES = {
    "hip_width": ("lhip", "rhip"),
    "ankle_distance": ("lankle", "rankle"),
    "hind_paws_distance": ("lhpaw", "rhpaw"),
    "shoulder_width": ("lshoulder", "rshoulder"),
    "front_paws_distance": ("lfpaw", "rfpaw"),
    "cheek_distance": ("lcheek", "rcheek"),
    "tailbase_tailtip_distance": ("tailbase", "tailtip"),
    "hip_tailbase_distance": ("hip", "tailbase"),
    "hip_sternumtail_distance": ("hip", "sternumtail"),
    "sternumtail_sternumhead_distance": ("sternumtail", "sternumhead"),
    "sternumhead_neck_distance": ("sternumhead", "neck"),
    "neck_snout_distance": ("neck", "snout"),
    "hind_left_toes_spread": ("lhpd1t", "lhpd5t"),
    "hind_right_toes_spread": ("rhpd1t", "rhpd5t"),
    "hind_left_paw_length": ("lankle", "lhpd3t"),
    "hind_right_paw_length": ("rankle", "rhpd3t"),
}


class BodyPartDistanceDef(Feature, SummaryColumn):
    def __init__(self, dest: str, part1: str, part2: str):
        self.dest = dest
        self.part1 = part1
        self.part2 = part2

    def extract(self, ctx: FeaturesContext):
        label = ctx.label
        ctx._data[self.dest] = body_parts_distance(label, self.part1, self.part2)

    def summarize(self, ctx):
        ctx._data[f"{self.dest} (pixel)"] = np.nanmean(ctx._features[self.dest])


VectorParts = Tuple[str, str]
AngleSign = Literal["positive", "negative"]

ANGLE_FEATURES: Dict[
    str, Tuple[VectorParts, VectorParts, AngleSign, Union[str, None]]
] = {
    "chest_head_angle": (
        ("neck", "snout"),
        ("sternumtail", "sternumhead"),
        "positive",
        None,
    ),
    "hip_chest_angle": (
        ("sternumtail", "sternumhead"),
        ("tailbase", "hip"),
        "positive",
        None,
    ),
    "tail_hip_angle": (
        ("tailbase", "hip"),
        ("tailtip", "tailbase"),
        "negative",
        None,
    ),
    "hip_tailbase_hlpaw_angle": (
        ("tailbase", "hip"),
        ("tailbase", "lhpaw"),
        "positive",
        None,
    ),
    "hip_tailbase_hrpaw_angle": (
        ("tailbase", "rhpaw"),
        ("tailbase", "hip"),
        "positive",
        None,
    ),
    "midline_hlpaw_angle": (
        ("tailbase", "sternumtail"),
        ("lankle", "lhpaw"),
        "positive",
        "hind_left_paw_angle",
    ),
    "midline_hrpaw_angle": (
        ("rankle", "rhpaw"),
        ("tailbase", "sternumtail"),
        "positive",
        "hind_right_paw_angle",
    ),
}


class BodyPartAngleDef(Feature, SummaryColumn):
    vector_parts_1: VectorParts
    vector_parts_2: VectorParts
    sign: AngleSign
    summary_dest: Union[str, None]

    def __init__(
        self,
        dest: str,
        vector_parts_1: VectorParts,
        vector_parts_2: VectorParts,
        sign: AngleSign,
        summary_dest: Union[str, None] = None,
    ):
        self.dest = dest
        self.vector_parts_1 = vector_parts_1
        self.vector_parts_2 = vector_parts_2
        self.sign = sign
        self.summary_dest = summary_dest

    def extract(self, ctx: FeaturesContext):
        label = ctx.label
        vector1 = get_vector(label, self.vector_parts_1[0], self.vector_parts_1[1])
        vector2 = get_vector(label, self.vector_parts_2[0], self.vector_parts_2[1])
        if self.sign == "positive":
            ctx._data[self.dest] = get_angle(vector1, vector2)
        elif self.sign == "negative":
            ctx._data[self.dest] = -get_angle(vector1, vector2)

    def summarize(self, ctx):
        dest = self.summary_dest if self.summary_dest is not None else self.dest

        ctx._data[f"{dest} (degree)"] = np.nanmean(ctx._features[self.dest])
