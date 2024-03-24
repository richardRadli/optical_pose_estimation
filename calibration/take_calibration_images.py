import cv2
import logging
import os

from config.config import CameraAndCalibrationConfig
from config.config_selector import camera_config
from utils.utils import create_dir, create_timestamp, setup_logger


class CalibrationImageCapture:
    def __init__(self):
        setup_logger()
        timestamp = create_timestamp()
        root_dir = camera_config().get("calibration_images")
        self.location = create_dir(root_dir, timestamp)

        self.cam_cfg = CameraAndCalibrationConfig().parse()
        self.cap = None
        self.capture_count = 0
        self.size_coefficients = self.cam_cfg.size_coefficients

        self.setup_camera()

    def setup_camera(self) -> None:
        """
        This function tries to open the camera, and set the parameters of the device.

        Returns:
            None
        """

        self.cap = cv2.VideoCapture(self.cam_cfg.camID, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            logging.error("Could not open camera device")
            exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_cfg.height)
        self.cap.set(cv2.CAP_PROP_SETTINGS, 0)

    def capture_images(self) -> None:
        """
        This function opens the stream, displays it on a window. If [c] is pressed, we save an image to a designated
        directory. If button [q] is pressed, the program shuts down.

        Returns:
             None
        """

        while True:
            _, frame = self.cap.read()

            resized_frame = cv2.resize(src=frame,
                                       dsize=(frame.shape[1] // self.size_coefficients,
                                              frame.shape[0] // self.size_coefficients))

            cv2.imshow('Calibration Image', resized_frame)
            key = cv2.waitKey(1)

            if key == ord("q"):
                break
            elif key == ord("c"):
                filename = f"calibration_image_{self.capture_count:03d}.png"
                filename_location = os.path.join(self.location, filename)
                cv2.imwrite(filename_location, frame)
                logging.info(f"Captured image: {filename}")
                self.capture_count += 1

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    calibration = CalibrationImageCapture()
    calibration.capture_images()
