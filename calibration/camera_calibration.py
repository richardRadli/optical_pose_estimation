import cv2
import logging
import numpy as np
import os

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config.config import CameraAndCalibrationConfig
from config.config_selector import camera_config
from utils.utils import create_dir, create_timestamp, file_reader, find_latest_subdir, setup_logger


class CameraCalibration:
    def __init__(self):
        setup_logger()
        timestamp = create_timestamp()
        self.camera_cfg = CameraAndCalibrationConfig().parse()

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, self.camera_cfg.square_size, 0.001)
        self.camera_matrix = np.empty([3, 3])
        self.undistorted_camera_matrix = np.empty([3, 3])
        self.dist_coefficients = np.empty([1, 5])
        self.roi = None
        self.rotation_vectors = None
        self.translation_vectors = None
        self.object_points = []
        self.image_points = []
        self.chs_col = self.camera_cfg.chs_col
        self.chs_row = self.camera_cfg.chs_row
        self.image_size = None
        self.original_images = []

        source_images = find_latest_subdir(camera_config().get("calibration_images"))
        self.original_images_list = file_reader(source_images, "jpg")
        self.save_data_path = create_dir(camera_config().get("camera_matrix"), timestamp)
        self.undistorted_images_path = create_dir(camera_config().get("undistorted_calibration_images"), timestamp)

    # ---------------------------------------------------------------------------------------------------------------- #
    # -------------------------------------------- F I N D   C O R N E R S ------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    def find_corners(self, thread_id: int, image_name: str):
        name = f"thr-{thread_id}"
        corner_results = None

        src = cv2.imread(image_name)
        image_name = os.path.basename(image_name)
        grey = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        if self.image_size is None:
            self.image_size = grey.shape[:2]

        ret, corners = cv2.findChessboardCorners(grey, (self.chs_col, self.chs_row), None, cv2.CALIB_CB_FAST_CHECK)
        image_data = image_name, src

        if not ret:
            logging.warning(f"{name}: Could not find corners for {image_name}")
        else:
            corner_results = cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), criteria=self.criteria)

        return corner_results, image_data

    def calibration(self) -> None:
        accepted = 0

        with ThreadPoolExecutor() as executor:
            futures = []

            for i, img in tqdm(enumerate(self.original_images_list),
                               total=len(self.original_images_list),
                               desc="Collecting images"):
                futures.append(executor.submit(self.find_corners, i, img))

            try:
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing calibration images"):
                    corners, img_data = future.result()
                    if corners is not None:
                        accepted += 1
                        self.image_points.append(corners)
                    self.original_images.append(img_data)
            except Exception as e:
                logging.error(str(e))

        rejected = len(self.original_images_list) - accepted
        logging.info(f'Accepted images: {accepted}')
        logging.info(f'Rejected images: {rejected}')

        objp = np.zeros((self.chs_row * self.chs_col, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chs_col, 0:self.chs_row].T.reshape(-1, 2)
        self.object_points = [objp] * len(self.image_points)

        logging.info('------C A L I B R A T I O N   S T A R T E D-----')
        rms, self.camera_matrix, self.dist_coefficients, self.rotation_vectors, self.translation_vectors = (
            cv2.calibrateCamera(objectPoints=self.object_points,
                                imagePoints=self.image_points,
                                imageSize=self.image_size,
                                cameraMatrix=None,
                                distCoeffs=None)
        )

        logging.info(f'Root mean square error\n{rms}')
        logging.info(f'Camera matrix\n{self.camera_matrix}')
        logging.info(f'Distortion coefficients\n{self.dist_coefficients}')
        logging.info('-----C A L I B R A T I O N   F I N I S H E D-----')

    # ---------------------------------------------------------------------------------------------------------------- #
    # --------------------------------------- U N D I S T O R T   A N D   S A V E ------------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #
    @staticmethod
    def undistort_and_save(name: str, image, matrix, distortion_coefficients, undistorted_matrix, roi,
                           undistorted_images_path: str) -> None:
        image = cv2.undistort(image, matrix, distortion_coefficients, None, undistorted_matrix)
        x, y, w, h = roi
        image = image[y:y + h, x:x + w]
        path = os.path.join(undistorted_images_path, os.path.basename(name))
        cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 3])

    # ---------------------------------------------------------------------------------------------------------------- #
    # ------------------------------- G E N E R A T E   N E W   C A M E R A   M A T R I X ---------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    def generate_new_camera_matrix(self) -> None:
        logging.info('----- O B T A I N I N G   N E W   C A M E R A   M A T R I X-----')
        name, img = self.original_images[0]
        height, width = img.shape[:2]
        self.undistorted_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix,
                                                                                 self.dist_coefficients,
                                                                                 (width, height),
                                                                                 1,
                                                                                 (width, height))

        logging.info('New camera matrix')
        logging.info(str(self.undistorted_camera_matrix))
        self.undistort_and_save(name=name,
                                image=img,
                                matrix=self.camera_matrix,
                                distortion_coefficients=self.dist_coefficients,
                                undistorted_matrix=self.undistorted_camera_matrix,
                                roi=self.roi,
                                undistorted_images_path=self.undistorted_images_path)

        mean_error = 0
        for i in range(len(self.object_points)):
            img_points2, _ = cv2.projectPoints(self.object_points[i],
                                               self.rotation_vectors[i],
                                               self.translation_vectors[i],
                                               self.camera_matrix,
                                               self.dist_coefficients)
            error = cv2.norm(self.image_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
            mean_error += error

        total_err = mean_error / len(self.object_points)
        logging.info(f'Total error: {total_err}')

        if total_err > self.camera_cfg.error_threshold:
            raise Exception('Camera calibration error too high!')

        path = os.path.join(self.save_data_path, "undistorted_cam_mtx.npz")
        np.savez(path,
                 camera_matrix=self.camera_matrix,
                 distortion_coefficients=self.dist_coefficients,
                 undistorted_camera_matrix=self.undistorted_camera_matrix,
                 roi=self.roi)
        logging.info(f"Saved camera calibration to {path}")

    # ---------------------------------------------------------------------------------------------------------------- #
    # ----------------------------------------- U N D I S T O R T   I M A G E S -------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    def undistort_calibration_images(self) -> None:
        """
        Undistorts every image used during calibration and saves them in the appropriate folder.
        """

        with ThreadPoolExecutor() as executor:
            futures = []
            for name, img in self.original_images[1:]:
                futures.append(executor.submit(self.undistort_and_save,
                                               name,
                                               img,
                                               self.camera_matrix,
                                               self.dist_coefficients,
                                               self.undistorted_camera_matrix,
                                               self.roi,
                                               self.undistorted_images_path))

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------- M A I N --------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    def main(self):
        self.calibration()
        self.generate_new_camera_matrix()
        if self.camera_cfg.undistort:
            self.undistort_calibration_images()


if __name__ == "__main__":
    try:
        cal = CameraCalibration()
        cal.main()
    except KeyboardInterrupt as kie:
        logging.error(f"{kie}")
