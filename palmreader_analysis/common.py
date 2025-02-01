"""
Most features have a 1:1 relationship with some column in the summary CSV. In those cases, it makes the most sense to define the computation for both the feature and the summarized column in the same place. This module has all of those columns.

Any features which do not have a single column in the summary are defined in the `features.py` file, and likely have one or more summary columns defined in `summary.py`. This separation is not visible to the user. To get all columns for either features or summary, use `FeaturesContext.get_all_features` or `SummaryContext.get_all_columns`.
"""

from .features import FeaturesContext, Feature
from .summary import NanSumColumn
from utils import cal_distance_


class DistanceDeltaDef(Feature, NanSumColumn):
    def __init__(self):
        NanSumColumn.__init__(self, "distance_delta", "distance_traveled (pixel)")

    def extract(self, ctx: FeaturesContext):
        ctx._data["distance_delta"] = cal_distance_(ctx.label).reshape(-1)
