from typing import Any, List, Dict
from .variants import Paw, LuminanceMeasure
from utils import *


class FeaturesContext:
    @staticmethod
    def get_all_features() -> List["Feature"]:
        # we have to import these here otherwise we get a circular import. same thing with the SummaryContext
        from .common import (
            DistanceDeltaDef,
            BodyPartDistanceDef,
            BodyPartAngleDef,
            DISTANCE_FEATURES,
            ANGLE_FEATURES,
        )

        features = []

        # add paw-specific features
        for paw in Paw:
            # add reworked luminance-based measures
            for kind in LuminanceMeasure:
                features.append(PawFeatureDef(paw, kind))

            # add legacy luminance measures
            features.append(LegacyPawLuminanceDef(paw))

        # add background luminance
        features.append(BackgroundLuminanceDef())

        # add single features
        features.append(SingleFeaturesDef())

        # add distance delta feature
        features.append(DistanceDeltaDef())

        # Add Animal detection features
        features.append(AnimalDetectionDef())

        # Add body parts distance features
        for column in DISTANCE_FEATURES.keys():
            part1, part2 = DISTANCE_FEATURES[column]
            features.append(BodyPartDistanceDef(column, part1, part2))

        # Add body part angle features
        for column in ANGLE_FEATURES.keys():
            vector_parts_1, vector_parts_2, sign, _summary_dest = ANGLE_FEATURES[column]
            features.append(
                BodyPartAngleDef(column, vector_parts_1, vector_parts_2, sign)
            )

        for paw in Paw:
            features.append(TrackingLikelihoodDef(paw))

        return features

    _data: Dict[str, pd.DataFrame]
    _cache: Dict[str, Any]

    def __init__(self, name, tracking_path, ftir_path):
        # Context provided to feature calculation functions
        self.name = name
        self.tracking_path = tracking_path
        self.ftir_path = ftir_path
        self.df = pd.read_hdf(tracking_path)
        self.model_id = self.df.columns[0][0]
        self.label = self.df[self.model_id]
        self.ftir_video = cv2.VideoCapture(ftir_path)

        # Data generated from feature calculation functions
        self._data = {}
        self._cache = {}

    def to_hdf5(self, dest_path):
        with h5py.File(dest_path, "w") as hdf:
            video_data = hdf.create_group(self.name)
            for key in self._data.keys():
                video_data.create_dataset(key, data=self._data[key])


class Feature:
    def extract(self, ctx: FeaturesContext):
        raise NotImplementedError


class PawLuminanceComputation:
    @staticmethod
    def compute_paw_luminance(ctx: FeaturesContext) -> PawLuminanceData:
        """
        Computes the paw luminance. If this computation has already been done, it is not repeated.
        """
        if "paw_luminance" not in ctx._cache:
            ctx._cache["paw_luminance"] = cal_paw_luminance_rework(
                ctx.label, ctx.ftir_video, size=22
            )

        return ctx._cache["paw_luminance"]


class PawFeatureDef(Feature):
    def __init__(self, paw: Paw, measure: LuminanceMeasure):
        self.paw = paw
        self.kind = measure

    def extract(self, ctx: FeaturesContext):
        luminance_data = PawLuminanceComputation.compute_paw_luminance(ctx)

        # add the paw feature to the dictionary
        ctx._data[f"{self.paw.value}_{self.kind.feature_name()}"] = (
            luminance_data.get_measure(self.kind)[self.paw.value]
        )


class LegacyPawLuminanceDef(Feature):
    def __init__(self, paw: Paw):
        self.paw = paw

    def extract(self, ctx: FeaturesContext):
        luminance_data = PawLuminanceComputation.compute_paw_luminance(ctx)

        paw_data = luminance_data.legacy_paw_luminance.get_paw(self.paw)

        # add the paw feature to the dictionary
        ctx._data[f"{self.paw.old_name()}_luminance"] = paw_data


class BackgroundLuminanceDef(Feature):
    def extract(self, ctx: FeaturesContext):
        luminance_data = PawLuminanceComputation.compute_paw_luminance(ctx)

        # add the background luminance to the dictionary
        ctx._data["background_luminance"] = luminance_data.background_luminance


class SingleFeaturesDef(Feature):
    """
    This is a special feature definition that computes multiple columns. It is the only feature that does this. It does this because it provides scalar values that are not optional and are only used internally.

    Palmreader users will never see these values directly.
    """

    def extract(self, ctx: FeaturesContext):
        luminance_data = PawLuminanceComputation.compute_paw_luminance(ctx)
        fps = int(ctx.ftir_video.get(cv2.CAP_PROP_FPS))

        # add the single features to the dictionary
        ctx._data["fps"] = np.array(fps)
        ctx._data["frame_count"] = np.array(luminance_data.frame_count)


class AnimalDetectionDef(Feature):
    def __init__(self):
        pass

    def extract(self, ctx: FeaturesContext):
        ctx._data["animal_detection"] = detect_animal_in_recording(
            ctx.label, ctx._data["fps"]
        )


class TrackingLikelihoodDef(Feature):
    def __init__(self, paw: Paw):
        self.paw = paw
        pass

    def extract(self, ctx: FeaturesContext):
        label = ctx.label
        ctx._data[f"{self.paw.value}_tracking_likelihood"] = label[self.paw.value][
            "likelihood"
        ]
