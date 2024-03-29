import cv2
import os
import logging
import numpy as np

from concurrent.futures import ThreadPoolExecutor

from config.config import CameraAndCalibrationConfig
from config.config_selector import camera_config
from utils.utils import (create_dir, create_timestamp, find_latest_file_in_latest_directory, find_latest_subdir,
                         setup_logger)


class UndistortImages:
    def __init__(self):
        setup_logger()
        timestamp = create_timestamp()
        self.camera_cfg = CameraAndCalibrationConfig().parse()

        camera_matrix_file = find_latest_file_in_latest_directory(path=camera_config().get("camera_matrix"))
        data = np.load(camera_matrix_file)

        self.camera_matrix = data["camera_matrix"]
        self.dist_coefficients = data["distortion_coefficients"]
        self.undistorted_camera_matrix = data["undistorted_camera_matrix"]
        self.roi = data["roi"]

        self.input_dir, output_dir = self.get_input_dir(self.camera_cfg.operation)
        self.output_dir = create_dir(output_dir, timestamp)

    @staticmethod
    def get_input_dir(operation):
        input_dir = {
            "calibration_images":
                find_latest_subdir(camera_config().get(operation)),
            "chessboard_images":
                find_latest_subdir(camera_config().get(operation)),
            "motherboard_images":
                find_latest_subdir(camera_config().get(operation))
        }

        if operation not in input_dir.keys():
            raise ValueError(f"Operation {operation} not in the input directory list!")

        output_dir = {
            "undistorted_calibration_images":
                camera_config().get(operation),
            "undistorted_chessboard_images":
                find_latest_subdir(camera_config().get(operation)),
            "undistorted_motherboard_images":
                find_latest_subdir(camera_config().get(operation))
        }

        if operation not in output_dir.keys():
            raise ValueError(f"Operation {operation} not in the output directory list!")

        return input_dir[operation], output_dir[operation]

    def process_image(self, image_path: str, output_path: str) -> None:
        image = cv2.imread(image_path)
        undistorted_image = cv2.undistort(src=image,
                                          cameraMatrix=self.camera_matrix,
                                          distCoeffs=self.dist_coefficients,
                                          dst=None,
                                          newCameraMatrix=self.undistorted_camera_matrix)

        x, y, w, h = self.roi
        undistorted_image = undistorted_image[y:y+h, x:x+w]
        cv2.imwrite(output_path, undistorted_image)
        logging.info(f"Image saved to {output_path}")

    def undistort_images(self) -> None:
        with ThreadPoolExecutor(max_workers=self.camera_cfg.cpu_threads) as executor:
            image_paths = [os.path.join(self.input_dir, filename) for filename in os.listdir(self.input_dir)]
            output_path = [os.path.join(self.output_dir, os.path.basename(path)) for path in image_paths]
            executor.map(self.process_image, image_paths, output_path)


if __name__ == "__main__":
    undst_images = UndistortImages()
    undst_images.undistort_images()
