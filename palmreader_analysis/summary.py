from typing import Any, List, Union, Dict, Tuple, Iterable
import pandas as pd
import numpy as np
from collections import defaultdict
import h5py
from cols_name_dicts import summary_col_name_dict
from utils import both_front_paws_lifted

from .variants import (
    Paw,
    LuminanceMeasure,
    RatioOrder,
    ColumnMetadata,
    ColumnCategory,
    all_paws_tags,
)


class SummaryContext:
    @staticmethod
    def get_all_columns() -> List["SummaryColumn"]:
        # we have to import these here otherwise we get a circular import. same thing with the FeaturesContext
        from .common import (
            DistanceDeltaDef,
            BodyPartDistanceDef,
            BodyPartAngleDef,
            DISTANCE_FEATURES,
            ANGLE_FEATURES,
        )

        columns = []

        # TODO: determine desired ordering of columns, maybe deduplicate some of these loops

        columns.append(TimeInformationColumns())

        columns.append(DistanceDeltaDef())

        for measure in LuminanceMeasure:
            columns.append(AverageOverallLuminanceColumn(measure))

            for paw in Paw:
                columns.append(AveragePawLuminanceColumn(paw, measure))
                columns.append(RelativePawLuminanceColumn(paw, measure))

            for ratio_order in RatioOrder:
                columns.append(HindPawRatioColumn(measure, ratio_order))
                columns.append(
                    HindPawRatioColumn(measure, ratio_order, StandingMaskComputer())
                )
            columns.append(FrontToHindPawRatioColumn(measure))

        for paw in Paw:
            columns.append(PawLiftedTimeColumn(paw))

        columns.append(BothFrontPawsLiftedColumn())

        for paw in Paw:
            columns.append(LegacyPawLuminanceColumn(paw))

        columns.append(LegacyAllPawsLuminanceColumn())

        for paw in Paw:
            columns.append(LegacyRelativePawLuminanceColumn(paw))

        for ratio_order in RatioOrder:
            columns.append(LegacyHindPawRatioColumn(ratio_order))
            columns.append(
                LegacyHindPawRatioColumn(ratio_order, LegacyStandingMaskComputer())
            )

        columns.append(LegacyFrontToHindRatioColumn())

        for paw in Paw:
            columns.append(LegacyPawLiftedTimeColumn(paw))

        columns.append(LegacyBothFrontPawsLiftedColumn())

        for column in DISTANCE_FEATURES.keys():
            part1, part2 = DISTANCE_FEATURES[column]
            columns.append(BodyPartDistanceDef(column, part1, part2))

        for column in ANGLE_FEATURES.keys():
            vector_parts_1, vector_parts_2, sign, dest = ANGLE_FEATURES[column]
            columns.append(
                BodyPartAngleDef(column, vector_parts_1, vector_parts_2, sign, dest)
            )

        for paw in Paw:
            columns.append(TrackingLikelihoodColumn(paw))

        # this column must be computed after the tracking likelihood columns
        # TODO: use a computation class for the likelihood?
        columns.append(QualityControlFlagColumn())

        return columns

    @staticmethod
    def merge_to_df(contexts: Iterable["SummaryContext"]) -> pd.DataFrame:
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

            self._features["frame_count"] = end_frame - start_frame

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

    def finish(self):
        # this just applies the name dictionary to the columns
        # TODO: remove this and just change the computation methods
        self._data = {
            summary_col_name_dict[key]: value for key, value in self._data.items()
        }


class SummaryColumn:
    def summarize(self, ctx: SummaryContext):
        raise NotImplementedError

    def metadata(self) -> List[ColumnMetadata]:
        raise NotImplementedError


class TimeInformationColumns(SummaryColumn):
    """
    This is a special column definition that computes multiple columns. It is the only column that does this. It does this because it provides binning-related information that is not strictly scientific and will be produced in every summary CSV.
    """

    TOTAL_RECORDING_TIME = "total recording_time (min)"
    BIN_START_END = "PV: bin start-end (min)"
    BIN_DURATION = "bin duration (min)"

    def summarize(self, ctx):
        ctx._data[TimeInformationColumns.TOTAL_RECORDING_TIME] = (
            ctx._features["frame_count"] / ctx._features["fps"] / 60
        )
        ctx._data[TimeInformationColumns.BIN_START_END] = (
            f"{ctx._start_time:.2f} - {ctx._end_time:.2f}"
        )
        ctx._data[TimeInformationColumns.BIN_DURATION] = ctx._end_time - ctx._start_time

    def metadata(self):
        return [
            ColumnMetadata.make_hidden(
                TimeInformationColumns.TOTAL_RECORDING_TIME, ColumnCategory.TEMPORAL
            ),
            ColumnMetadata.make_hidden(
                TimeInformationColumns.BIN_START_END, ColumnCategory.TEMPORAL
            ),
            ColumnMetadata.make_hidden(
                TimeInformationColumns.BIN_DURATION, ColumnCategory.TEMPORAL
            ),
        ]


class PawLuminanceMeanDataHolder:
    _data: Dict[Paw, Dict[LuminanceMeasure, np.ndarray]]

    def __init__(self):
        self._data = defaultdict(dict)

    def set_value(self, paw: Paw, measure: LuminanceMeasure, value: float):
        self._data[paw][measure] = value

    def get_value(self, paw: Paw, measure: LuminanceMeasure) -> float:
        return self._data[paw][measure]

    def get_sum(self, measure: LuminanceMeasure) -> float:
        measure_sum = 0

        for paw in Paw:
            measure_sum += self.get_value(paw, measure)

        return measure_sum


class Mask:
    NONE: "Mask"

    def __init__(self, name: str, mask: np.ndarray):
        self.name = name
        self.mask = mask

    def apply(self, original: Any) -> Any:
        """
        Original must be mask-able using `[]` indexing.
        """
        # if we don't have a name, we must be the NONE mask
        if self.name == "":
            return original
        else:
            return original[self.mask]

    def cache_key(self, original: str) -> str:
        if self.name == "":
            return original
        else:
            return f"{original}_{self.name}"


Mask.NONE = Mask("", np.array([]))


class PawLuminanceMeanComputation:
    @staticmethod
    def compute_paw_luminance_average(
        ctx: SummaryContext, mask: Mask = Mask.NONE
    ) -> PawLuminanceMeanDataHolder:
        key = mask.cache_key("paw_luminance")

        if key not in ctx._cache:
            paw_luminance = PawLuminanceMeanDataHolder()

            for paw in Paw:
                for measure in LuminanceMeasure:
                    paw_luminance.set_value(
                        paw,
                        measure,
                        np.nanmean(
                            mask.apply(
                                ctx._features[f"{paw.value}_{measure.feature_name()}"]
                            )
                        ),
                    )

            ctx._cache[key] = paw_luminance

        return ctx._cache[key]


class AverageOverallLuminanceColumn(SummaryColumn):
    def __init__(self, measure: LuminanceMeasure):
        self.measure = measure

    def _get_column_name(self) -> str:
        return f"average_overall_{self.measure.value} ({self.measure.units()})"

    def summarize(self, ctx):
        paw_luminance = PawLuminanceMeanComputation.compute_paw_luminance_average(ctx)

        ctx._data[self._get_column_name()] = paw_luminance.get_sum(self.measure)

    def metadata(self):
        return [
            ColumnMetadata.make(
                column=self._get_column_name(),
                category=ColumnCategory.LUMINANCE_BASED,
                tags=all_paws_tags(),
                displayname=f"Average overall {self.measure.displayname()}",
                description=f"The sum of the average {self.measure.displayname()} of all paws",
            )
        ]


class AveragePawLuminanceColumn(SummaryColumn):
    def __init__(self, paw: Paw, measure: LuminanceMeasure):
        self.paw = paw
        self.measure = measure

    def _get_column_name(self) -> str:
        return f"average_{self.paw.old_name()}_{self.measure.value} ({self.measure.units()})"

    def summarize(self, ctx):
        paw_luminance = PawLuminanceMeanComputation.compute_paw_luminance_average(ctx)

        ctx._data[self._get_column_name()] = paw_luminance.get_value(
            self.paw, self.measure
        )

    def metadata(self):
        return [
            ColumnMetadata.make(
                column=self._get_column_name(),
                category=ColumnCategory.LUMINANCE_BASED,
                tags=[self.paw.as_tag()],
                displayname=f"Average {self.paw.displayname()} {self.measure.displayname()}",
                description=f"The average {self.measure.displayname()} of the {self.paw.displayname()}",
            )
        ]


class RelativePawLuminanceColumn(SummaryColumn):
    def __init__(self, paw: Paw, measure: LuminanceMeasure):
        self.paw = paw
        self.measure = measure

    def _get_column_name(self) -> str:
        return f"relative_{self.paw.old_name()}_{self.measure.value} (ratio)"

    def summarize(self, ctx):
        paw_luminance = PawLuminanceMeanComputation.compute_paw_luminance_average(ctx)

        ctx._data[self._get_column_name()] = paw_luminance.get_value(
            self.paw, self.measure
        ) / paw_luminance.get_sum(self.measure)

    def metadata(self):
        return [
            ColumnMetadata.make(
                column=self._get_column_name(),
                category=ColumnCategory.LUMINANCE_BASED,
                tags=[self.paw.as_tag()],
                displayname=f"Relative {self.paw.displayname()} {self.measure.displayname()}",
                description=f"The ratio of the average {self.measure.displayname()} of the {self.paw.displayname()} to the sum of the average {self.measure.displayname()} of all paws",
            )
        ]


class MaskComputer:
    NONE: "MaskComputer"

    def __init__(self, name: str):
        self.name = name

    def compute(self, ctx: SummaryContext) -> Mask:
        raise NotImplementedError

    def column_infix(self) -> str:
        if self.name == "":
            return ""
        else:
            return f"_{self.name}"

    def displayname(self) -> str:
        if self.name == "":
            return ""
        else:
            return f" {self.name}"

    def description_infix(self) -> str:
        if self.name == "":
            return ""
        else:
            # lack of space after when is intentional, displayname adds it
            return f" when{self.displayname()}"


class NoneMaskComputer(MaskComputer):
    def compute(self, ctx: SummaryContext) -> Mask:
        return Mask.NONE


MaskComputer.NONE = NoneMaskComputer("")


class StandingMaskComputer(MaskComputer):
    def __init__(self):
        super().__init__("standing")

    def compute(self, ctx: SummaryContext) -> Mask:
        if "standing_mask" not in ctx._cache:
            standing_mask = both_front_paws_lifted(
                ctx._features[f"{Paw.LEFT_FRONT.value}_luminance_rework"],
                ctx._features[f"{Paw.RIGHT_FRONT.value}_luminance_rework"],
            )
            ctx._cache["standing_mask"] = Mask("standing", standing_mask)

        return ctx._cache["standing_mask"]


class HindPawRatioColumn(SummaryColumn):
    ratio_order: RatioOrder

    def __init__(
        self,
        measure: LuminanceMeasure,
        ratio_order: RatioOrder,
        mask: MaskComputer = MaskComputer.NONE,
    ):
        self.measure = measure
        self.ratio_order = ratio_order
        self.mask = mask

    def _get_column_name(self) -> str:
        return f"average{self.mask.column_infix()}_hind_paw_{self.measure.value}_ratio ({self.ratio_order.summary_display()})"

    def summarize(self, ctx):
        mask = self.mask.compute(ctx)

        paw_luminance = PawLuminanceMeanComputation.compute_paw_luminance_average(
            ctx, mask
        )

        left = paw_luminance.get_value(Paw.LEFT_HIND, self.measure)
        right = paw_luminance.get_value(Paw.RIGHT_HIND, self.measure)

        ratio = self.ratio_order.divide(left=left, right=right)

        ctx._data[self._get_column_name()] = ratio

    def metadata(self):
        return [
            ColumnMetadata.make(
                column=self._get_column_name(),
                category=ColumnCategory.LUMINANCE_BASED,
                tags=[Paw.LEFT_HIND.as_tag(), Paw.RIGHT_HIND.as_tag()],
                displayname=f"Average{self.mask.displayname()} {self.ratio_order.displayname()} hind paw {self.measure.displayname()} ratio",
                description=f"The average ratio of {self.measure.displayname()}{self.mask.description_infix()} of the left and right hind paws, divided {self.ratio_order.displayname()}",
            )
        ]


class FrontToHindPawRatioColumn(SummaryColumn):
    def __init__(self, measure: LuminanceMeasure):
        self.measure = measure

    def _get_column_name(self) -> str:
        return f"average_front_to_hind_paw_{self.measure.value}_ratio"

    def summarize(self, ctx):
        paw_luminance = PawLuminanceMeanComputation.compute_paw_luminance_average(ctx)

        left_front = paw_luminance.get_value(Paw.LEFT_FRONT, self.measure)
        right_front = paw_luminance.get_value(Paw.RIGHT_FRONT, self.measure)
        left_hind = paw_luminance.get_value(Paw.LEFT_HIND, self.measure)
        right_hind = paw_luminance.get_value(Paw.RIGHT_HIND, self.measure)

        front = left_front + right_front
        hind = left_hind + right_hind

        ctx._data[self._get_column_name()] = front / hind

    def metadata(self):
        return [
            ColumnMetadata.make(
                column=self._get_column_name(),
                category=ColumnCategory.LUMINANCE_BASED,
                tags=all_paws_tags(),
                displayname=f"Average front to hind paw {self.measure.displayname()} ratio",
                description=f"The average ratio of {self.measure.displayname()} of the front paws to the hind paws",
            )
        ]


class PawLiftedTimeColumn(SummaryColumn):
    def __init__(self, paw: Paw):
        self.paw = paw

    def _get_column_name(self) -> str:
        return f"{self.paw.old_name()}_paw_lifted_time (seconds)"

    def summarize(self, ctx):
        ctx._data[self._get_column_name()] = (
            np.sum(
                ctx._features[
                    f"{self.paw.value}_{LuminanceMeasure.LUMINANCE.feature_name()}"
                ]
                < 1e-4
            )
            / ctx._features["fps"]
        )

    def metadata(self):
        return [
            ColumnMetadata.make(
                column=self._get_column_name(),
                category=ColumnCategory.TEMPORAL,
                tags=[self.paw.as_tag()],
                displayname=f"{self.paw.displayname().capitalize()} paw lifted time",
                description=f"The amount of time the {self.paw.displayname()} paw was lifted (in seconds)",
            )
        ]


class BothFrontPawsLiftedColumn(SummaryColumn):
    COLUMN_NAME = "both_front_paws_lifted (seconds)"

    def summarize(self, ctx):
        standing = StandingMaskComputer().compute(ctx)
        standing = standing.mask

        ctx._data[BothFrontPawsLiftedColumn.COLUMN_NAME] = (
            np.sum(standing) / ctx._features["fps"]
        )

    def metadata(self):
        return [
            ColumnMetadata.make(
                column=BothFrontPawsLiftedColumn.COLUMN_NAME,
                category=ColumnCategory.TEMPORAL,
                tags=[
                    Paw.LEFT_FRONT.as_tag(),
                    Paw.RIGHT_FRONT.as_tag(),
                ],
                displayname=f"Both front paws lifted time",
                description=f"The amount of time both front paws were lifted (in seconds)",
            )
        ]


class LegacyPawLuminanceDataHolder:
    _data: Dict[Paw, float]
    _sum: float

    def __init__(self):
        self._data = defaultdict(float)

    def set_value(self, paw: Paw, value: float):
        self._data[paw] = value

    def get_value(self, paw: Paw) -> float:
        return self._data[paw]

    def set_sum(self, value: float):
        self._sum = value

    def get_sum(self) -> float:
        return self._sum


class LegacyStandingMaskComputer(MaskComputer):
    def __init__(self):
        super().__init__("standing")

    def compute(self, ctx: SummaryContext) -> Mask:
        if "legacy_standing_mask" not in ctx._cache:
            standing_mask = both_front_paws_lifted(
                ctx._features[f"{Paw.LEFT_FRONT.old_name()}_luminance"],
                ctx._features[f"{Paw.RIGHT_FRONT.old_name()}_luminance"],
            )
            ctx._cache["legacy_standing_mask"] = Mask("standing", standing_mask)

        return ctx._cache["legacy_standing_mask"]


class LegacyPawLuminanceComputation:
    @staticmethod
    def compute_paw_luminance_average(
        ctx: SummaryContext, mask: Mask = Mask.NONE
    ) -> LegacyPawLuminanceDataHolder:
        key = mask.cache_key("legacy_paw_luminance")

        if key not in ctx._cache:
            paw_luminance = LegacyPawLuminanceDataHolder()

            luminance_sum = 0

            for paw in Paw:
                value = np.nanmean(
                    mask.apply(ctx._features[f"{paw.old_name()}_luminance"])
                )

                paw_luminance.set_value(paw, value)

                luminance_sum += value

            paw_luminance.set_sum(luminance_sum)

            ctx._cache[key] = paw_luminance

        return ctx._cache[key]


class LegacyPawLuminanceColumn(SummaryColumn):
    def __init__(self, paw: Paw):
        self.paw = paw

    def _get_column_name(self) -> str:
        return f"legacy: average_{self.paw.old_name()}_luminance"

    def summarize(self, ctx):
        paw_luminance = LegacyPawLuminanceComputation.compute_paw_luminance_average(ctx)

        ctx._data[self._get_column_name()] = paw_luminance.get_value(self.paw)

    def metadata(self):
        return [
            ColumnMetadata.make(
                column=self._get_column_name(),
                category=ColumnCategory.LUMINANCE_BASED,
                tags=[self.paw.as_tag()],
                displayname=f"Legacy average {self.paw.displayname()} luminance",
                description=f"The average luminance of the {self.paw.displayname()}",
                legacy=True,
            )
        ]


class LegacyAllPawsLuminanceColumn(SummaryColumn):
    COLUMN_NAME = "legacy: average_all_paws_sum_luminance"

    def summarize(self, ctx):
        paw_luminance = LegacyPawLuminanceComputation.compute_paw_luminance_average(ctx)

        ctx._data[LegacyAllPawsLuminanceColumn.COLUMN_NAME] = paw_luminance.get_sum()

    def metadata(self):
        return [
            ColumnMetadata.make(
                column=LegacyAllPawsLuminanceColumn.COLUMN_NAME,
                category=ColumnCategory.LUMINANCE_BASED,
                tags=all_paws_tags(),
                displayname=f"Legacy average all paws sum luminance",
                description=f"The sum of the average luminance of all paws",
                legacy=True,
            )
        ]


class LegacyRelativePawLuminanceColumn(SummaryColumn):
    def __init__(self, paw: Paw):
        self.paw = paw

    def _get_column_name(self) -> str:
        return f"legacy: relative_{self.paw.old_name()}_luminance"

    def summarize(self, ctx):
        paw_luminance = LegacyPawLuminanceComputation.compute_paw_luminance_average(ctx)

        ctx._data[self._get_column_name()] = (
            paw_luminance.get_value(self.paw) / paw_luminance.get_sum()
        )

    def metadata(self):
        return [
            ColumnMetadata.make(
                column=self._get_column_name(),
                category=ColumnCategory.LUMINANCE_BASED,
                tags=[self.paw.as_tag()],
                displayname=f"Legacy relative {self.paw.displayname()} luminance",
                description=f"The ratio of the average luminance of the {self.paw.displayname()} to the sum of the average luminance of all paws",
                legacy=True,
            )
        ]


class LegacyHindPawRatioColumn(SummaryColumn):
    ratio_order: RatioOrder

    def __init__(
        self,
        ratio_order: RatioOrder,
        mask: MaskComputer = MaskComputer.NONE,
    ):
        self.ratio_order = ratio_order
        self.mask = mask

    def _get_column_name(self) -> str:
        return f"legacy: average{self.mask.column_infix()}_hind_paw_luminance_ratio ({self.ratio_order.summary_display()})"

    def summarize(self, ctx):
        mask = self.mask.compute(ctx)

        paw_luminance = LegacyPawLuminanceComputation.compute_paw_luminance_average(
            ctx, mask
        )

        left = paw_luminance.get_value(Paw.LEFT_HIND)
        right = paw_luminance.get_value(Paw.RIGHT_HIND)

        ratio = self.ratio_order.divide(left=left, right=right)

        ctx._data[self._get_column_name()] = ratio

    def metadata(self):
        return [
            ColumnMetadata.make(
                column=self._get_column_name(),
                category=ColumnCategory.LUMINANCE_BASED,
                tags=[Paw.LEFT_HIND.as_tag(), Paw.RIGHT_HIND.as_tag()],
                displayname=f"Legacy average{self.mask.displayname()} {self.ratio_order.displayname()} hind paw luminance ratio",
                description=f"The average ratio of luminance{self.mask.description_infix()} of the left and right hind paws, divided {self.ratio_order.displayname()}",
                legacy=True,
            )
        ]


class LegacyFrontToHindRatioColumn(SummaryColumn):
    COLUMN_NAME = "legacy: average_front_to_hind_paw_luminance_ratio"

    def summarize(self, ctx):
        paw_luminance = LegacyPawLuminanceComputation.compute_paw_luminance_average(ctx)

        left_front = paw_luminance.get_value(Paw.LEFT_FRONT)
        right_front = paw_luminance.get_value(Paw.RIGHT_FRONT)
        left_hind = paw_luminance.get_value(Paw.LEFT_HIND)
        right_hind = paw_luminance.get_value(Paw.RIGHT_HIND)

        front = left_front + right_front
        hind = left_hind + right_hind

        ctx._data[LegacyFrontToHindRatioColumn.COLUMN_NAME] = front / hind

    def metadata(self):
        return [
            ColumnMetadata.make(
                column=LegacyFrontToHindRatioColumn.COLUMN_NAME,
                category=ColumnCategory.LUMINANCE_BASED,
                tags=all_paws_tags(),
                displayname=f"Legacy average front to hind paw luminance ratio",
                description=f"The average ratio of luminance of the front paws to the hind paws",
                legacy=True,
            )
        ]


class LegacyPawLiftedTimeColumn(SummaryColumn):
    def __init__(self, paw: Paw):
        self.paw = paw

    def _get_column_name(self) -> str:
        return f"legacy: {self.paw.old_name()}_paw_lifted_time (seconds)"

    def summarize(self, ctx):
        ctx._data[self._get_column_name()] = (
            np.sum(ctx._features[f"{self.paw.old_name()}_luminance"] < 1e-4)
            / ctx._features["fps"]
        )

    def metadata(self):
        return [
            ColumnMetadata.make(
                column=self._get_column_name(),
                category=ColumnCategory.TEMPORAL,
                tags=[self.paw.as_tag()],
                displayname=f"Legacy {self.paw.displayname()} paw lifted time",
                description=f"The amount of time the {self.paw.displayname()} paw was lifted (in seconds)",
                legacy=True,
            )
        ]


class LegacyBothFrontPawsLiftedColumn(SummaryColumn):
    COLUMN_NAME = "legacy: both_front_paws_lifted (ratio of time)"

    def summarize(self, ctx):
        standing = LegacyStandingMaskComputer().compute(ctx)
        standing = standing.mask

        ctx._data[LegacyBothFrontPawsLiftedColumn.COLUMN_NAME] = np.nanmean(standing)

    def metadata(self):
        return [
            ColumnMetadata.make(
                column=LegacyBothFrontPawsLiftedColumn.COLUMN_NAME,
                category=ColumnCategory.TEMPORAL,
                tags=[
                    Paw.LEFT_FRONT.as_tag(),
                    Paw.RIGHT_FRONT.as_tag(),
                ],
                displayname=f"Legacy both front paws lifted time",
                description=f"The ratio of time both front paws were lifted",
                legacy=True,
            )
        ]


class TrackingLikelihoodColumn(SummaryColumn):
    def __init__(self, paw: Paw):
        self.paw = paw

    def _get_column_name(self) -> str:
        return f"average_{self.paw.value}_tracking_likelihood"

    def summarize(self, ctx):
        ctx._data[self._get_column_name()] = np.nanmean(
            ctx._features[f"{self.paw.value}_tracking_likelihood"]
        )

    def metadata(self):
        # todo: consider adding another category e.g. misc? not really temporal
        return [
            ColumnMetadata.make_hidden(self._get_column_name(), ColumnCategory.TEMPORAL)
        ]


class QualityControlFlagColumn(SummaryColumn):
    """
    **This column must be computed after the tracking likelihood columns.**
    """

    COLUMN_NAME = "paws_tracking_quality_control_flag"

    def summarize(self, ctx):

        ctx._data[QualityControlFlagColumn.COLUMN_NAME] = 0

        if (
            ctx._data[f"average_{Paw.LEFT_HIND.value}_tracking_likelihood"] < 0.4
            or ctx._data[f"average_{Paw.RIGHT_HIND.value}_tracking_likelihood"] < 0.4
            or ctx._data[f"average_{Paw.LEFT_FRONT.value}_tracking_likelihood"] < 0.6
            or ctx._data[f"average_{Paw.RIGHT_FRONT.value}_tracking_likelihood"] < 0.6
        ):
            ctx._data[QualityControlFlagColumn.COLUMN_NAME] = 1

    def metadata(self):
        return [
            ColumnMetadata.make_hidden(
                QualityControlFlagColumn.COLUMN_NAME, ColumnCategory.TEMPORAL
            )
        ]
