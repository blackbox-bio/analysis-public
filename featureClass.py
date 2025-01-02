from pkg_resources import non_empty_lines

from utils import *





class FeaturesContext:
    _data: dict[str, pd.DataFrame]
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

    def to_hdf5(self, dest_path):
        with h5py.File(dest_path, "w") as hdf:
            video_data = hdf.create_group(self.name)
            for featureSet in self._data:
                for key in self._data[featureSet]:
                    video_data.create_dataset(key, data=self._data[featureSet][key])

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



class PawFeaturesDef(FeatureDef):
    def __init__(self):
        pass

    def compute(self, ctx: FeaturesContext):
        # Feature Extraction
        (paw_luminescence,
         paw_print_size,
         paw_luminance,
         background_luminance,
         frame_count,
         legacy_paw_luminance) = cal_paw_luminance_rework(
            ctx.label, ctx.ftir_video, size=22
        )
        (hind_left,
         hind_right,
         front_left,
         front_right,) = legacy_paw_luminance
        #Paw Features
        pawFeatures = pd.DataFrame()
        pawFeatures["lhpaw_luminescence"] = paw_luminescence['lhpaw']
        pawFeatures["lhpaw_print_size"] = paw_print_size['lhpaw']
        pawFeatures["lhpaw_luminance_rework"] = paw_luminance['lhpaw']
        pawFeatures["hind_left_luminance"] = hind_left

        pawFeatures["lfpaw_luminescence"] = paw_luminescence['lfpaw']
        pawFeatures["lfpaw_print_size"] = paw_print_size['lfpaw']
        pawFeatures["lfpaw_luminance_rework"] = paw_luminance['lfpaw']
        pawFeatures["front_left_luminance"] = front_left

        pawFeatures["rfpaw_luminescence"] = paw_luminescence['rfpaw']
        pawFeatures["rfpaw_print_size"] = paw_print_size['rfpaw']
        pawFeatures["rfpaw_luminance_rework"]= paw_luminance['rfpaw']
        pawFeatures["front_right_luminance"] = front_right

        pawFeatures["rhpaw_luminescence"] = paw_luminescence['rhpaw']
        pawFeatures["rhpaw_print_size"] = paw_print_size['rhpaw']
        pawFeatures["rhpaw_luminance_rework"] = paw_luminance['rhpaw']
        pawFeatures["hind_right_luminance"] = hind_right

        pawFeatures["background_luminance"] = background_luminance
        ctx._data["singleFeatures"] = pd.DataFrame()
        ctx._data["singleFeatures"]["frame_count"] = np.array(frame_count)
        ctx._data["paw_features"] = pawFeatures

class AnimalDetectionDef(FeatureDef):
    def __init__(self):
        pass

    def compute(self, ctx: FeaturesContext):
        AnimalDetectionFeatures = pd.DataFrame()
        _fps = int(ctx.ftir_video.get(cv2.CAP_PROP_FPS))
        ctx._data["singleFeatures"]["fps"] = np.array(_fps)
        AnimalDetectionFeatures["animal_detection"] = detect_animal_in_recording(ctx.label, _fps)
        ctx._data["animal_detection_features"] = AnimalDetectionFeatures



class DistanceDeltaDef(FeatureDef):
    def __init__(self):
        pass

    def compute(self, ctx: FeaturesContext):
        DistanceDeltaFeatures = pd.DataFrame()
        DistanceDeltaFeatures["distance_delta"] = cal_distance_(ctx.label).reshape(-1)
        ctx._data["distance_delta_features"] = DistanceDeltaFeatures

class BodyPartDistanceDef(FeatureDef):
    def __init__(self):
        pass

    def compute(self, ctx: FeaturesContext):
        bodyPartsDistanceFeatures = pd.DataFrame()
        label = ctx.label

        bodyPartsDistanceFeatures["hip_width"] = body_parts_distance(label, "lhip", "rhip")
        bodyPartsDistanceFeatures["ankle_distance"] = body_parts_distance(label, "lankle", "rankle")
        bodyPartsDistanceFeatures["hind_paws_distance"] = body_parts_distance(label, "lhpaw", "rhpaw")
        # upper body parts distance
        bodyPartsDistanceFeatures["shoulder_width"] = body_parts_distance(label, "lshoulder", "rshoulder")
        bodyPartsDistanceFeatures["front_paws_distance"] = body_parts_distance(label, "lfpaw", "rfpaw")
        # face parts distance
        bodyPartsDistanceFeatures["cheek_distance"] = body_parts_distance(label, "lcheek", "rcheek")

        # midline body parts distance
        bodyPartsDistanceFeatures["tailbase_tailtip_distance"] = body_parts_distance(
            label, "tailbase", "tailtip"
        )
        bodyPartsDistanceFeatures["hip_tailbase_distance"] = body_parts_distance(label, "hip", "tailbase")
        bodyPartsDistanceFeatures["hip_sternumtail_distance"] = body_parts_distance(
            label, "hip", "sternumtail"
        )
        bodyPartsDistanceFeatures["sternumtail_sternumhead_distance"] = body_parts_distance(
            label, "sternumtail", "sternumhead"
        )
        bodyPartsDistanceFeatures["sternumhead_neck_distance"] = body_parts_distance(
            label, "sternumhead", "neck"
        )
        bodyPartsDistanceFeatures["neck_snout_distance"] = body_parts_distance(label, "neck", "snout")

        # toe spread and paw length for both hind paws
        bodyPartsDistanceFeatures["hind_left_toes_spread"] = body_parts_distance(label, "lhpd1t", "lhpd5t")
        bodyPartsDistanceFeatures["hind_right_toes_spread"] = body_parts_distance(label, "rhpd1t", "rhpd5t")
        bodyPartsDistanceFeatures["hind_left_paw_length"] = body_parts_distance(label, "lankle", "lhpd3t")
        bodyPartsDistanceFeatures["hind_right_paw_length"] = body_parts_distance(label, "rankle", "rhpd3t")

        ctx._data["bodyparts_distance_features"] = bodyPartsDistanceFeatures


class BodyPartAngleDef(FeatureDef):
    def __init__(self):
        pass

    def compute(self, ctx: FeaturesContext):
        bodyPartsAngleFeatures = pd.DataFrame()
        label = ctx.label

        sternumtail_sternumhead_vector = get_vector(label, "sternumtail", "sternumhead")
        midline_vector = get_vector(label, "tailbase", "sternumtail")
        neck_snout_vector = get_vector(label, "neck", "snout")
        tailbase_hip_vector = get_vector(label, "tailbase", "hip")
        tailtip_tailbase_vector = get_vector(label, "tailtip", "tailbase")
        tailbase_hlpaw_vec = get_vector(label, "tailbase", "lhpaw")
        tailbase_hrpaw_vec = get_vector(label, "tailbase", "rhpaw")
        lankle_lhpaw_vec = get_vector(label, "lankle", "lhpaw")
        # lankle_lhpd1t_vec = get_vector(label, "lankle", "lhpd1t")
        # lankle_lhpd3t_vec = get_vector(label,"lankle","lhpd3t")
        # lankle_lhpd5t_vec = get_vector(label, "lankle", "lhpd5t")
        rankle_rhpaw_vec = get_vector(label, "rankle", "rhpaw")
        # rankle_rhpd1t_vec = get_vector(label, "rankle", "rhpd1t")
        # rankle_rhpd3t_vec = get_vector(label,"rankle","rhpd3t")
        # rankle_rhpd5t_vec = get_vector(label, "rankle", "rhpd5t")

        # body parts angles
        bodyPartsAngleFeatures["chest_head_angle"] = get_angle(
            neck_snout_vector, sternumtail_sternumhead_vector
        )
        bodyPartsAngleFeatures["hip_chest_angle"] = get_angle(
            sternumtail_sternumhead_vector, tailbase_hip_vector
        )
        # note the negative sign for the tail_hip_angle
        bodyPartsAngleFeatures["tail_hip_angle"] = -get_angle(
            tailbase_hip_vector, tailtip_tailbase_vector
        )
        bodyPartsAngleFeatures["hip_tailbase_hlpaw_angle"] = get_angle(
            tailbase_hip_vector, tailbase_hlpaw_vec
        )
        bodyPartsAngleFeatures["hip_tailbase_hrpaw_angle"] = get_angle(
            tailbase_hrpaw_vec, tailbase_hip_vector
        )
        # paw angles with respect to the midline for both hind paws
        bodyPartsAngleFeatures["midline_hlpaw_angle"] = get_angle(midline_vector, lankle_lhpaw_vec)
        bodyPartsAngleFeatures["midline_hrpaw_angle"] = get_angle(rankle_rhpaw_vec, midline_vector)

        ctx._data["bodyparts_angle_features"] = bodyPartsAngleFeatures



class TrackingLikelyHoodDef(FeatureDef):
    def __init__(self):
        pass

    def compute(self, ctx: FeaturesContext):
        TrackingLikelyhoodFeatures = pd.DataFrame()
        label = ctx.label
        TrackingLikelyhoodFeatures["lhpaw_tracking_likelihood"] = label["lhpaw"]["likelihood"]
        TrackingLikelyhoodFeatures["rhpaw_tracking_likelihood"] = label["rhpaw"]["likelihood"]
        TrackingLikelyhoodFeatures["lfpaw_tracking_likelihood"] = label["lfpaw"]["likelihood"]
        TrackingLikelyhoodFeatures["rfpaw_tracking_likelihood"] = label["rfpaw"]["likelihood"]

        ctx._data["tracking_likelyhood_features"] = TrackingLikelyhoodFeatures
