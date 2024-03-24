from typing import Dict

from config.const import DATA_PATH, IMAGES_PATH


def camera_config() -> Dict:
    cam_config = {
        "calibration_images":
            IMAGES_PATH.get_data_path("calibration_images"),
        "chessboard_images":
            IMAGES_PATH.get_data_path("chessboard_images"),
        "motherboard_images":
            IMAGES_PATH.get_data_path("motherboard_images"),
        "camera_matrix":
            DATA_PATH.get_data_path("camera_matrix"),
        "camera_settings":
            DATA_PATH.get_data_path("camera_settings"),
        "undistorted_images":
            IMAGES_PATH.get_data_path("undistorted_images")

    }

    return cam_config
