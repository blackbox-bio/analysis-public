from typing import Any, List, Union, Dict
import pandas as pd
import numpy as np
from collections import defaultdict
import h5py
from .variants import Paw, LuminanceMeasure
from cols_name_dicts import summary_col_name_dict


class SummaryContext:
    @staticmethod
    def get_all_columns() -> List["SummaryColumn"]:
        from .common import DistanceDeltaDef

        columns = []

        columns.append(DistanceDeltaDef())

        for measure in LuminanceMeasure:
            columns.append(AverageOverallLuminanceColumn(measure))

            for paw in Paw:
                columns.append(AveragePawLuminanceColumn(paw, measure))
                columns.append(RelativePawLuminanceColumn(paw, measure))

        return columns

    @staticmethod
    def merge_contexts(contexts: List["SummaryContext"]) -> Dict[str, Dict[str, Any]]:
        summary_features = {}

        for context in contexts:
            summary_features[context.name] = {
                # apply column renaming here. TODO: remove this and just change
                # the computation methods
                summary_col_name_dict[k]: v
                for k, v in context._data.items()
            }

        return summary_features

    @staticmethod
    def merge_to_df(contexts: List["SummaryContext"]) -> pd.DataFrame:
        summary_features = SummaryContext.merge_contexts(contexts)

        return pd.DataFrame.from_dict(summary_features, orient="index")

    @staticmethod
    def compare_summary_columns(contexts: List["SummaryContext"], old_df: pd.DataFrame):
        error_count = 0
        pass_count = 0
        missing_count = 0

        new_df = SummaryContext.merge_to_df(contexts)

        print(new_df)
        print(old_df)

        for video in old_df.index:
            print(f"{video}")
            old_video = old_df.loc[video]
            new_video = new_df.loc[video]
            for key in old_video.keys():
                if key in new_video:
                    if np.isnan(new_video[key]) and np.isnan(old_video[key]):
                        pass_count += 1
                        print(f"{video}: PASS: '{key}' is NaN in both new and old")
                    elif new_video[key] == old_video[key]:
                        pass_count += 1
                        print(f"{video}: PASS: '{key}' is the same in both new and old")
                    else:
                        error_count += 1
                        print(f"{video}: FAIL: '{key}' is different in new and old")
                else:
                    missing_count += 1
                    print(f"{video}: MISSING: '{key}' is missing in new")

        print("Summary columns comparison:")
        print(f"PASS: {pass_count}")
        print(f"FAIL: {error_count}")
        print(f"MISSING: {missing_count}")

    _data: Dict[str, Any]
    _features: Dict[str, Union[h5py.Group, h5py.Dataset, h5py.Datatype, Any]]
    _cache: Dict[str, Any]

    def __init__(self, features_file: str):
        self._data = {}
        self._features = {}
        self._cache = {}

        # load the features file
        with h5py.File(features_file, "r") as hdf:
            # this loop will only run once as each h5 file only has one
            # top-level key
            for video in hdf.keys():
                self.name = video
                for feature in hdf[video].keys():
                    self._features[feature] = np.array(hdf[video][feature])


class SummaryColumn:
    def summarize(self, ctx: SummaryContext):
        raise NotImplementedError


class NanSumColumn(SummaryColumn):
    def __init__(self, feature: str, dest: str):
        self.feature = feature
        self.dest = dest

    def summarize(self, ctx):
        ctx._data[self.dest] = np.nansum(ctx._features[self.feature])


class PawLuminanceMeanDataHolder:
    _data: Dict[Paw, Dict[LuminanceMeasure, np.ndarray]]

    def __init__(self):
        self._data = defaultdict(dict)

    def set_value(self, paw: Paw, measure: LuminanceMeasure, value: np.ndarray):
        self._data[paw][measure] = value

    def get_value(self, paw: Paw, measure: LuminanceMeasure) -> np.ndarray:
        return self._data[paw][measure]

    def get_sum(self, measure: LuminanceMeasure) -> float:
        measure_sum = 0

        for paw in Paw:
            measure_sum += self.get_value(paw, measure)

        return measure_sum


class PawLuminanceMeanComputation:
    @staticmethod
    def compute_paw_luminance_average(
        ctx: SummaryContext,
    ) -> PawLuminanceMeanDataHolder:
        if "paw_luminance" not in ctx._cache:
            paw_luminance = PawLuminanceMeanDataHolder()

            for paw in Paw:
                for measure in LuminanceMeasure:
                    paw_luminance.set_value(
                        paw,
                        measure,
                        np.nanmean(
                            ctx._features[f"{paw.value}_{measure.feature_name()}"]
                        ),
                    )

            ctx._cache["paw_luminance"] = paw_luminance

        return ctx._cache["paw_luminance"]


class AverageOverallLuminanceColumn(SummaryColumn):
    def __init__(self, measure: LuminanceMeasure):
        self.measure = measure

    def summarize(self, ctx):
        paw_luminance = PawLuminanceMeanComputation.compute_paw_luminance_average(ctx)

        ctx._data[f"average_overall_{self.measure.value} ({self.measure.units()})"] = (
            paw_luminance.get_sum(self.measure)
        )


class AveragePawLuminanceColumn(SummaryColumn):
    def __init__(self, paw: Paw, measure: LuminanceMeasure):
        self.paw = paw
        self.measure = measure

    def summarize(self, ctx):
        paw_luminance = PawLuminanceMeanComputation.compute_paw_luminance_average(ctx)

        ctx._data[
            f"average_{self.paw.old_name()}_{self.measure.value} ({self.measure.units()})"
        ] = paw_luminance.get_value(self.paw, self.measure)


class RelativePawLuminanceColumn(SummaryColumn):
    def __init__(self, paw: Paw, measure: LuminanceMeasure):
        self.paw = paw
        self.measure = measure

    def summarize(self, ctx):
        paw_luminance = PawLuminanceMeanComputation.compute_paw_luminance_average(ctx)

        ctx._data[f"relative_{self.paw.old_name()}_{self.measure.value} (ratio)"] = (
            paw_luminance.get_value(self.paw, self.measure)
            / paw_luminance.get_sum(self.measure)
        )
