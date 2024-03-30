import logging
import os

from utils.utils import setup_logger


class _Const(object):
    setup_logger()
    user = os.getlogin()

    root_mapping = {
        "rrb12": {
            "DATA_ROOT":
                "C:/Users/rrb12/Documents/project/storage"
        }
    }

    if user not in root_mapping:
        raise ValueError("User name {user} not found in root_mapping.".format(user=user))

    root_info = root_mapping[user]
    DATA_ROOT = root_info["DATA_ROOT"]

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- C R E A T E   D I R C T O R I E S ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def create_directories(cls, dirs, root_type):
        for _, path, in dirs.items():
            if root_type == "DATA_ROOT":
                dir_path = os.path.join(cls.DATA_ROOT, path)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                    logging.info(f"Directory {dir_path} has been created")
            else:
                raise ValueError("Root type {root_type} not found.".format(root_type=root_type))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++ I M A G E S +++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Images(_Const):
    dir_images = {
        "calibration_images":
            "images/camera/calibration_images",

        "chessboard_images":
            "images/camera/chessboard_images",

        "motherboard_images":
            "images/camera/motherboard_images",

        "undistorted_calibration_images":
            "images/objects/calibration_images",

        "undistorted_chessboard_images":
            "images/objects/chessboard_images",

        "undistorted_motherboard_images":
            "images/objects/motherboard_images",

        "roi_images":
            "images/pose_estimation/roi_images",

        "pairing_images":
            "images/pose_estimation/pairing_images",

        "object_camera_cord_sys":
            "images/pose_estimation/object_camera_cord_sys",

        "object_camera_world_sys":
            "images/pose_estimation/object_camera_world_sys"
    }

    def __init__(self):
        super().__init__()
        self.create_directories(self.dir_images, "DATA_ROOT")

    def get_data_path(self, key):
        return os.path.join(self.DATA_ROOT, self.dir_images.get(key, ""))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++ D A T A +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Data(_Const):
    dir_data = {
        "camera_matrix":
            "data/camera/camera_matrix",

        "camera_settings":
            "data/camera/camera_settings",

        "model_coordinates":
            "data/model_coordinates",

        "mounting_hole_center_points":
            "data/pose_estimation/mounting_hole_center_points"
    }

    def __init__(self):
        super().__init__()
        self.create_directories(self.dir_data, "DATA_ROOT")

    def get_data_path(self, key):
        return os.path.join(self.DATA_ROOT, self.dir_data.get(key, ""))


CONST = _Const()
IMAGES_PATH: Images = Images()
DATA_PATH: Data = Data()
