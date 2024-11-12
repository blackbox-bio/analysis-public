from pkg_resources import non_empty_lines

from utils import *





class FeatureClass:
    def __init__(self,tracking_path,ftir_path):
        # Overhead for tracking
        self.tracking_path = tracking_path
        self.ftir_path = ftir_path
        # Feature data identification
        self.df = pd.read_hdf(tracking_path)
        self.model_id = self.df.columns[0][0]
        self.label = self.df[self.model_id]
        self.ftir_video = cv2.VideoCapture(ftir_path)
        self.fps = int(self.ftir_video.get(cv2.CAP_PROP_FPS))

        # Feature Calculation
        self.distance_delta = cal_distance_(self.label).reshape(-1)

        self.lhpaw, self.lfpaw,self.rhpaw,self.rfpaw, self.background_lumiance,self.frame_count = PawFeaturesClass().create_paws(self.label,self.ftir_video)





class PawFeaturesClass:
    def __init__(self):
        self.luminescence = None
        self.print_size = None
        self.luminance = None
        self.luminance_legacy = None


    def create_paws(self,label,ftir_video):
        # Feature Extraction
        (paw_luminescence,
         paw_print_size,
         paw_luminance,
         background_luminance,
         frame_count,
         legacy_paw_luminance) = cal_paw_luminance_rework(
            label, ftir_video, size=22
        )
        (hind_left,
         hind_right,
         front_left,
         front_right,) = legacy_paw_luminance
        # Specific Paw Features
        lhpaw = PawFeaturesClass()
        lhpaw.luminescence= paw_luminescence['lhpaw']
        lhpaw.print_size= paw_print_size['lhpaw']
        lhpaw.luminance = paw_luminance['lhpaw']
        lhpaw.luminance_legacy = hind_left

        lfpaw = PawFeaturesClass()
        lfpaw.luminescence= paw_luminescence['lfpaw']
        lfpaw.print_size= paw_print_size['lfpaw']
        lfpaw.luminance = paw_luminance['lfpaw']
        lfpaw.luminance_legacy = front_left

        rfpaw = PawFeaturesClass()
        rfpaw.luminescence= paw_luminescence['rfpaw']
        rfpaw.print_size= paw_print_size['rfpaw']
        rfpaw.luminance = paw_luminance['rfpaw']
        rfpaw.luminance_legacy = front_right

        rhpaw = PawFeaturesClass()
        rhpaw.luminescence= paw_luminescence['rhpaw']
        rhpaw.print_size= paw_print_size['rhpaw']
        rhpaw.luminance = paw_luminance['rhpaw']
        rhpaw.luminance_legacy = hind_right

        return lhpaw, lfpaw, rhpaw, rfpaw, background_luminance, frame_count


