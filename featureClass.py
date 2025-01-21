from pkg_resources import non_empty_lines
from typing import Any, Literal, List
from utils import *
from enum import Enum




class FeaturesContext:
    _data: dict[str, pd.DataFrame]
    _cache: dict[str, Any]

    @staticmethod
    def get_all_features() -> List["FeatureDef"]:
        features = []

        # add paw-specific features
        for paw in Paw:
            # add reworked luminance-based measures
            for kind in ["luminescence", "print_size", "luminance_rework"]:
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
        distanceWithParts = {
            "hip_width": ["lhip", "rhip"],
            "ankle_distance": ["lankle", "rankle"],
            "hind_paws_distance": ["lhpaw", "rhpaw"],
            "shoulder_width": ["lshoulder", "rshoulder"],
            "front_paws_distance": ["lfpaw", "rfpaw"],
            "cheek_distance": ["lcheek", "rcheek"],
            "tailbase_tailtip_distance": ["tailbase", "tailtip"],
            "hip_tailbase_distance": ["hip", "tailbase"],
            "hip_sternumtail_distance": ["hip", "sternumtail"],
            "sternumtail_sternumhead_distance": ["sternumtail", "sternumhead"],
            "sternumhead_neck_distance": ["sternumhead", "neck"],
            "neck_snout_distance": ["neck", "snout"],
            "hind_left_toes_spread": ["lhpd1t", "lhpd5t"],
            "hind_right_toes_spread": ["rhpd1t", "rhpd5t"],
            "hind_left_paw_length": ["lankle", "lhpd3t"],
            "hind_right_paw_length": ["rankle", "rhpd3t"]
        }
        for column in distanceWithParts.keys():
            partInfo = distanceWithParts[column]
            features.append(BodyPartDistanceDef(column,partInfo[0],partInfo[1]))


        # Add body part angle features
        anglesFromVectorsWithParts = {
            "chest_head_angle" : [["neck","snout"],["sternumtail","sternumhead"],"positive"],
            "hip_chest_angle" : [["sternumtail","sternumhead"],["tailbase","hip"],"positive"],
            "tail_hip_angle" : [["tailbase","hip"],["tailtip","tailbase"],"negative"],
            "hip_tailbase_hlpaw_angle" : [["tailbase","hip"],["tailbase","lhpaw"],"positive"],
            "hip_tailbase_hrpaw_angle" : [["tailbase","rhpaw"],["tailbase","hip"],"positive"],
            "midline_hlpaw_angle" : [["tailbase","sternumtail"],["lankle","lhpaw"],"positive"],
            "midline_hrpaw_angle" : [["rankle","rhpaw"],["tailbase","sternumtail"],"positive"]
        }
        for column in anglesFromVectorsWithParts.keys():
            angleInfo = anglesFromVectorsWithParts[column]
            features.append(BodyPartAngleDef(column,angleInfo[0],angleInfo[1],angleInfo[2]))

        for paw in Paw:
            features.append(TrackingLikelyHoodDef(paw))

        return features

        # TODO: add the rest of the features after porting them to the new system

    def __init__(self,name,tracking_path,ftir_path):
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

    def FeatureClassTestNew(self, oldFeatureDict):
        # the new implementations don't have the top level keys so we can just compare each key directly
        errorCount = 0
        passCount = 0

        # Cant directly compare every feature with just ==, so convert them to series for access to .equals
        for key in oldFeatureDict:
            if pd.Series(oldFeatureDict[key]).equals(pd.Series(self._data[key])):
                print(f"PASS: {key} was found equal in both dataframes")
                passCount += 1
            else:
                print(f"ERROR: {key} was found NOT equal in both dataframes")
                errorCount += 1

        print(f"{errorCount}/{passCount} features had missing or incorrect data")
        print(f"{passCount}/{len(oldFeatureDict)} were migrated")

    def FeatureClassTest(self, oldFeatureDict):
        errorCount = 0
        featureCount = 0
        missing = oldFeatureDict.keys()
        paw_features = self._data["paw_features"].columns.tolist()
        for feature in paw_features:
            if pd.Series(oldFeatureDict[feature]).equals(self._data["paw_features"][feature]):
                print(f"PASS: {feature} was found equal in both dataframes")
                missing = list(set(missing) - set(feature))
                featureCount += 1
            else:
                print(f"ERROR: {feature} was found NOT equal in both dataframes")
                errorCount += 1

        bodyparts_distance_features = self._data["bodyparts_distance_features"].columns.tolist()
        for feature in bodyparts_distance_features:
            if oldFeatureDict[feature].equals(self._data["bodyparts_distance_features"][feature]):
                print(f"PASS: {feature} was found equal in both dataframes")
                missing = list(set(missing) - set(feature))
                featureCount += 1
            else:
                print(f"ERROR: {feature} was found NOT equal in both dataframes")
                errorCount += 1

        animal_detection_features = self._data["animal_detection_features"].columns.tolist()
        for feature in animal_detection_features:
            if pd.Series(oldFeatureDict[feature]).equals(self._data["animal_detection_features"][feature]):
                print(f"PASS: {feature} was found equal in both dataframes")
                missing = list(set(missing) - set(feature))
                featureCount += 1
            else:
                print(f"ERROR: {feature} was found NOT equal in both dataframes")
                errorCount += 1

        bodyparts_angle_features = self._data["bodyparts_angle_features"].columns.tolist()
        for feature in bodyparts_angle_features:
            if pd.Series(oldFeatureDict[feature]).equals(self._data["bodyparts_angle_features"][feature]):
                print(f"PASS: {feature} was found equal in both dataframes")
                missing = list(set(missing) - set(feature))
                featureCount += 1
            else:
                print(f"ERROR: {feature} was found NOT equal in both dataframes")
                errorCount += 1

        tracking_likelyhood_features = self._data["tracking_likelyhood_features"].columns.tolist()
        for feature in tracking_likelyhood_features:
            if pd.Series(oldFeatureDict[feature]).equals(self._data["tracking_likelyhood_features"][feature]):
                print(f"PASS: {feature} was found equal in both dataframes")
                missing = list(set(missing) - set(feature))
                featureCount += 1
            else:
                print(f"ERROR: {feature} was found NOT equal in both dataframes")
                errorCount += 1

        if pd.Series(oldFeatureDict["distance_delta"]).equals(self._data["distance_delta_features"]["distance_delta"]):
            print(f"PASS: distance_delta was found equal in both dataframes")
            featureCount += 1
        else:
            print(f"ERROR: distance_delta was found NOT equal in both dataframes")
            errorCount += 1

        if pd.Series(oldFeatureDict["fps"]).equals(self._data["singleFeatures"]["fps"]):
            print(f"PASS: fps was found equal in both dataframes")
            featureCount += 1
        else:
            print(f"ERROR: fps was found NOT equal in both dataframes")
            errorCount += 1

        if pd.Series(oldFeatureDict["frame_count"]).equals(self._data["singleFeatures"]["frame_count"]):
            print(f"PASS: frame count was found equal in both dataframes")
            featureCount += 1
        else:
            print(f"ERROR: frame count was found NOT equal in both dataframes")
            errorCount += 1

        print(f"{errorCount}/{featureCount} features had missing or incorrect data")
        print(f"{featureCount}/{len(oldFeatureDict)} were migrated")


class FeatureDef:
    def compute(self, ctx: FeaturesContext):
        pass

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

class PawLuminanceComputation:
    @staticmethod
    def compute_paw_luminance(ctx: FeaturesContext):
        """
        Computes the paw luminance. If this computation has already been done, it is not repeated.
        """
        if 'paw_luminance' not in ctx._cache:
            ctx._cache['paw_luminance'] = cal_paw_luminance_rework(
                ctx.label, ctx.ftir_video, size=22
            )
        
        return ctx._cache['paw_luminance']

class PawFeatureDef(FeatureDef):
    def __init__(self, paw: Paw, kind: Literal["luminescence", "print_size", "luminance_rework"]):
        self.paw = paw
        self.kind = kind

    def compute(self, ctx: FeaturesContext):
        (paw_luminescence, paw_print_size, paw_luminance,_,_,_) = PawLuminanceComputation.compute_paw_luminance(ctx)
        
        # map the corresponding dictionaries to the provided `kind`
        luminance_data = {
            "luminescence": paw_luminescence,
            "print_size": paw_print_size,
            "luminance_rework": paw_luminance
        }
        
        # add the paw feature to the dictionary
        ctx._data[f"{self.paw.value}_{self.kind}"] = luminance_data[self.kind][self.paw.value]

class LegacyPawLuminanceDef(FeatureDef):
    def __init__(self, paw: Paw):
       self.paw = paw
    
    def compute(self, ctx: FeaturesContext):
        (_, _, _, _, _, legacy_paw_luminance) = PawLuminanceComputation.compute_paw_luminance(ctx)

        (hind_left,
         hind_right,
         front_left,
         front_right) = legacy_paw_luminance

        # map the luminance data to the corresponding paw
        luminance_data = {
            Paw.LEFT_HIND: hind_left,
            Paw.RIGHT_HIND: hind_right,
            Paw.LEFT_FRONT: front_left,
            Paw.RIGHT_FRONT: front_right
        }
        
        # add the paw feature to the dictionary
        ctx._data[f"{self.paw.old_name()}_luminance"] = luminance_data[self.paw]

class BackgroundLuminanceDef(FeatureDef):
    def compute(self, ctx: FeaturesContext):
        (_, _, _, background_luminance, _, _) = PawLuminanceComputation.compute_paw_luminance(ctx)
        
        # add the background luminance to the dictionary
        ctx._data["background_luminance"] = background_luminance

class SingleFeaturesDef(FeatureDef):
    """
    This is a special feature definition that computes multiple columns. It is the only feature that does this. It does this because it provides scalar values that are not optional and are only used internally.

    Palmreader users will never see these values directly.
    """
    def compute(self, ctx: FeaturesContext):
        (_, _, _, _, frame_count, _) = PawLuminanceComputation.compute_paw_luminance(ctx)
        fps = int(ctx.ftir_video.get(cv2.CAP_PROP_FPS))

        # add the single features to the dictionary
        ctx._data["fps"] = np.array(fps)
        ctx._data["frame_count"] = np.array(frame_count)


# TODO: port the following features to the new system
class AnimalDetectionDef(FeatureDef):
    def __init__(self):
        pass

    def compute(self, ctx: FeaturesContext):
        ctx._data["animal_detection"] = detect_animal_in_recording(ctx.label, ctx._data["fps"])


class DistanceDeltaDef(FeatureDef):
    def __init__(self):
        pass

    def compute(self, ctx: FeaturesContext):
        ctx._data["distance_delta"] = cal_distance_(ctx.label).reshape(-1)

class BodyPartDistanceDef(FeatureDef):
    def __init__(self, dest:str, part1:str, part2:str):
        self.dest = dest
        self.part1 = part1
        self.part2 = part2
        pass

    def compute(self, ctx: FeaturesContext):
        label = ctx.label
        ctx._data[self.dest] = body_parts_distance(label,self.part1,self.part2)


class BodyPartAngleDef(FeatureDef):
    def __init__(self,dest:str, vectorParts1:List, vectorParts2:List, sign=str):
        self.dest = dest
        self.vectorParts1 = vectorParts1
        self.vectorParts2 = vectorParts2
        self.sign = sign
        pass

    def compute(self, ctx: FeaturesContext):
        label = ctx.label
        vector1 = get_vector(label, self.vectorParts1[0], self.vectorParts1[1])
        vector2 = get_vector(label, self.vectorParts2[0], self.vectorParts2[1])
        if self.sign == "positive":
            ctx._data[self.dest] = get_angle(vector1,vector2)
        elif self.sign == "negative":
            ctx._data[self.dest] = -get_angle(vector1,vector2)



class TrackingLikelyHoodDef(FeatureDef):
    def __init__(self, paw:Paw):
        self.paw = paw
        pass

    def compute(self, ctx: FeaturesContext):
        label = ctx.label
        ctx._data[f"{self.paw.value}_tracking_likelihood"] = label[self.paw.value]["likelihood"]
