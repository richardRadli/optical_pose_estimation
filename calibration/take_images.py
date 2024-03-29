import cv2
import logging
import os

from config.config import CameraAndCalibrationConfig
from config.config_selector import camera_config
from utils.utils import create_dir, create_timestamp, setup_logger


class ImageCapture:
    def __init__(self):
        setup_logger()
        timestamp = create_timestamp()

        calibration_images_root_dir = camera_config().get("calibration_images")
        self.calibration_images_location = create_dir(calibration_images_root_dir, timestamp)

        chessboard_images_root_dir = camera_config().get("chessboard_images")
        self.chessboard_images_location = create_dir(chessboard_images_root_dir, timestamp)

        motherboard_images_root_dir = camera_config().get("motherboard_images")
        self.motherboard_images_location = create_dir(motherboard_images_root_dir, timestamp)

        self.cam_cfg = CameraAndCalibrationConfig().parse()
        self.cap = None

        self.capture_count_calibration = 0
        self.capture_count_chessboard = 0
        self.capture_count_motherboard = 0

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

    @staticmethod
    def capture_images(counter: int, frame: any, images_location: str, file_name_string: str) -> int:
        """
        Capture and save images.

        Args:
            counter (int): The current image counter.
            frame (any): The frame to capture the image from.
            images_location (str): The directory where images will be saved.
            file_name_string (str): The base filename for the captured images.

        Returns:
            int: The updated image counter.
        """

        filename = f"{file_name_string}_{counter:03d}.png"
        filename_location = os.path.join(images_location, filename)
        cv2.imwrite(filename_location, frame)
        logging.info(f"Captured image: {filename}")
        counter += 1
        return counter

    def main(self) -> None:
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
                self.capture_count_calibration = self.capture_images(self.capture_count_calibration,
                                                                     frame,
                                                                     self.calibration_images_location,
                                                                     "calibration_images")
            elif key == ord("b"):
                self.capture_count_chessboard = self.capture_images(self.capture_count_chessboard,
                                                                    frame,
                                                                    self.chessboard_images_location,
                                                                    "chessboard_images")
            elif key == ord("m"):
                self.capture_count_motherboard = self.capture_images(self.capture_count_motherboard,
                                                                     frame,
                                                                     self.motherboard_images_location,
                                                                     "mother_images")

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    calibration = ImageCapture()
    calibration.main()
