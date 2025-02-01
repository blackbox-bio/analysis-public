from typing import Any, List, Union, Dict, Tuple
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

        columns.append(TimeInformationColumns())

        columns.append(DistanceDeltaDef())

        for measure in LuminanceMeasure:
            columns.append(AverageOverallLuminanceColumn(measure))

            for paw in Paw:
                columns.append(AveragePawLuminanceColumn(paw, measure))
                columns.append(RelativePawLuminanceColumn(paw, measure))

        return columns

    @staticmethod
    def merge_to_df(contexts: List["SummaryContext"]) -> pd.DataFrame:
        summary_features = {}

        for context in contexts:
            summary_features[context.name] = context._data

        return pd.DataFrame.from_dict(summary_features, orient="index")

    _data: Dict[str, Any]
    _features: Dict[str, Union[h5py.Group, h5py.Dataset, h5py.Datatype, Any]]
    _cache: Dict[str, Any]

    _start_time: float
    _end_time: float

    def __init__(self, features_file: str, time_bin: Tuple[float, float]):
        self._data = {}
        self._features = {}
        self._cache = {}

        # load the features file
        with h5py.File(features_file, "r") as hdf:
            if len(hdf.keys()) != 1:
                raise ValueError(
                    f"Features file '{features_file}' has more than one top-level key"
                )

            self.name = list(hdf.keys())[0]
            for feature in hdf[self.name].keys():
                self._features[feature] = np.array(hdf[self.name][feature])

        # apply animal detection
        if "animal_detection" in self._features.keys():
            animal_detection = self._features["animal_detection"]
            frame_count = self._features["frame_count"]
            fps = self._features["fps"]

            start_frame = 0
            end_frame = frame_count

            for i in range(frame_count):
                if animal_detection[i] == 1:
                    start_frame = i
                    break

            self._apply_bin(start_frame, end_frame)

        # apply binning
        frame_count = self._features["frame_count"]
        fps = self._features["fps"]

        bin_start, bin_end = time_bin

        start_frame = int(bin_start * 60 * fps)

        if bin_end == -1:
            end_frame = frame_count
        else:
            end_frame = int(bin_end * 60 * fps)

        # Palmreader does not allow any of these conditions to happen
        if start_frame >= frame_count:
            raise ValueError(
                f"Invalid time bin: bin start is after the end of the video"
            )
        if start_frame < 0:
            start_frame = 0
        if end_frame > frame_count:
            end_frame = frame_count
        if end_frame < 0:
            raise ValueError(
                f"Invalid time bin: bin end is before the start of the video"
            )
        if end_frame <= start_frame:
            raise ValueError(
                f"Invalid time bin: bin end is before the start of the bin (bin duration is zero)"
            )

        self._apply_bin(start_frame, end_frame)

        # N.B. the old version stored these in the features object but we don't
        # need to do that
        self._start_time = start_frame / fps / 60
        self._end_time = end_frame / fps / 60

    def _apply_bin(self, start_frame: int, end_frame: int):
        for key in self._features.keys():
            # these two columns are scalar values
            if key in ["frame_count", "fps"]:
                continue

            self._features[key] = self._features[key][start_frame:end_frame]

            self._features["frame_count"] = end_frame - start_frame

    def finish(self):
        # this just applies the name dictionary to the columns
        # TODO: remove this and just change the computation methods
        self._data = {
            summary_col_name_dict[key]: value for key, value in self._data.items()
        }

    def compare_summary_columns(self, old_video: Dict[Any, Any]):
        error_count = 0
        pass_count = 0
        missing_count = 0

        for key in old_video.keys():
            if key in self._data:
                old_value = old_video[key]
                new_value = self._data[key]
                if new_value == old_value:
                    pass_count += 1
                    print(f"PASS: '{key}' is the same in both new and old")
                else:
                    error_count += 1
                    print(
                        f"FAIL: '{key}' is different in new and old. new = {new_value}, old = {old_value}"
                    )
            else:
                missing_count += 1
                print(f"MISSING: '{key}' is missing in new")

        print("Summary columns comparison:")
        print(f"PASS: {pass_count}")
        print(f"FAIL: {error_count}")
        print(f"MISSING: {missing_count}")


class SummaryColumn:
    def summarize(self, ctx: SummaryContext):
        raise NotImplementedError


class TimeInformationColumns(SummaryColumn):
    """
    This is a special column definition that computes multiple columns. It is the only column that does this. It does this because it provides binning-related information that is not strictly scientific and will be produced in every summary CSV.
    """

    def summarize(self, ctx):
        ctx._data["total recording_time (min)"] = (
            ctx._features["frame_count"] / ctx._features["fps"] / 60
        )
        ctx._data["PV: bin start-end (min)"] = (
            f"{ctx._start_time:.2f} - {ctx._end_time:.2f}"
        )
        ctx._data["bin duration (min)"] = ctx._end_time - ctx._start_time


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
