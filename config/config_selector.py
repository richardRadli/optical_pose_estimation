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
        "undistorted_calibration_images":
            IMAGES_PATH.get_data_path("undistorted_calibration_images"),
        "undistorted_chessboard_images":
            IMAGES_PATH.get_data_path("undistorted_chessboard_images"),
        "undistorted_motherboard_images":
            IMAGES_PATH.get_data_path("undistorted_motherboard_images"),
        "camera_matrix":
            DATA_PATH.get_data_path("camera_matrix"),
        "camera_settings":
            DATA_PATH.get_data_path("camera_settings")
    }

    return cam_config


def pose_estimation_config() -> Dict:
    pose_est = {
        "roi_images":
            IMAGES_PATH.get_data_path("roi_images"),
        "pairing_images":
            IMAGES_PATH.get_data_path("pairing_images"),
        "mounting_hole_center_points":
            DATA_PATH.get_data_path("mounting_hole_center_points")
    }

    return pose_est
