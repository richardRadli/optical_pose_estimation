import cv2
import json
import logging
import numpy as np
import os
import pandas as pd

from numpy import dot
from typing import Any, Dict, List, Optional, Tuple

from config.config import PoseEstimationConfig
from config.config_selector import camera_config, feature_matching_config, pose_estimation_config
from utils.utils import create_dir, create_timestamp, find_latest_file_in_latest_directory, setup_logger


class PoseEstimation:
    def __init__(self):
        timestamp = create_timestamp()
        pose_est_cfg = PoseEstimationConfig().parse()
        setup_logger()
        np.set_printoptions(suppress=True, precision=4)

        undistorted_camera_mtx_path = (
            find_latest_file_in_latest_directory(camera_config().get("camera_matrix"))
        )
        motherboard_image_path = (
            find_latest_file_in_latest_directory(camera_config().get("undistorted_motherboard_images"))
        )
        chessboard_image_path = (
            find_latest_file_in_latest_directory(camera_config().get("undistorted_chessboard_images"))
        )
        mounting_holes_data_path = (
            find_latest_file_in_latest_directory(feature_matching_config().get("mounting_hole_center_points"))
        )
        chessboard_model_points_path = (
            pose_estimation_config().get("chessboard_model_points")
        )
        motherboard_model_points_path = (
            pose_estimation_config().get("motherboard_model_points")
        )

        data = np.load(undistorted_camera_mtx_path)
        self.undistorted_camera_mtx = data["undistorted_camera_matrix"]
        self.motherboard_image = cv2.imread(motherboard_image_path, cv2.IMREAD_COLOR)
        self.chessboard_image = cv2.imread(chessboard_image_path, cv2.IMREAD_COLOR)

        with open(mounting_holes_data_path, "r") as file:
            mounting_holes_data = json.load(file)
        self.mounting_holes_data = self.json_to_np_array(mounting_holes_data)

        with open(chessboard_model_points_path, "r") as file:
            chessboard_model_points = json.load(file)

        with open(motherboard_model_points_path, "r") as file:
            motherboard_model_points = json.load(file)

        self.motherboard_model_points = (
            self.get_motherboard_coordinates(motherboard_model_points, pose_est_cfg.motherboard_type)
        )

        self.chessboard_model_points, self.num_rows, self.num_cols = (
            self.get_chessboard_coordinates(chessboard_model_points, pose_est_cfg.chessboard_type)
        )

        # Variables
        self.translation_vector_chessboard = None
        self.rotation_vector_chessboard = None
        self.rotation_matrix_chessboard = None
        self.rotation_inv_matrix_chessboard = None
        self.xc = None
        self.translation_vector_object = None
        self.rotation_vector_object = None
        self.rotation_matrix_object = None
        self.object_image_points = None
        self.xw = None
        self.world_coordinates = None
        self.number_of_points = len(self.motherboard_model_points)
        self.dist_coefficients = np.zeros((4, 1))

        self.object_camera_cord_sys_path = (
            create_dir(pose_estimation_config().get("object_camera_cord_sys"), timestamp)
        )
        self.object_camera_world_sys_path = (
            create_dir(pose_estimation_config().get("object_camera_world_sys"), timestamp)
        )

    @staticmethod
    def json_to_np_array(coordinates_list: List[List[float]]) -> np.ndarray:
        """
        Converts a list of coordinates into a NumPy array.

        Args:
            coordinates_list (List[List[float]]): A list of coordinate lists, where each inner list represents
                                                    a coordinate [x, y, z].

        Returns:
            np.ndarray: A NumPy array containing the coordinates.
        """

        coordinates_array = np.array(coordinates_list, dtype=np.float32)
        return coordinates_array

    @staticmethod
    def get_motherboard_coordinates(data: Dict[str, Any], motherboard_name: str) -> Optional[np.ndarray]:
        """
        Extracts coordinates of a specific motherboard from the given data.

        Args:
            data (Dict[str, Any]): A dictionary containing motherboard data.
            motherboard_name (str): The name of the motherboard to retrieve coordinates for.

        Returns:
            Optional[np.ndarray]: A NumPy array containing the coordinates of the specified motherboard,
                                  or None if the motherboard is not found.
        """

        motherboard_coordinates = None
        for motherboard in data["motherboards"]:
            if motherboard["name"] == motherboard_name:
                motherboard_coordinates = motherboard["coordinates"]
                break

        if motherboard_coordinates:
            coordinates_list = []
            for coord in motherboard_coordinates:
                x = coord["x"]
                y = coord["y"]
                z = coord["z"]
                coordinates_list.append((x, y, z))
            return np.array(coordinates_list, dtype=np.float32)
        else:
            return None

    @staticmethod
    def get_chessboard_coordinates(data: Dict[str, Any], chessboard_name: str) -> Tuple[np.ndarray, int, int]:
        """
        Extracts coordinates of a specific chessboard from the given data.

        Args:
            data (Dict[str, Any]): A dictionary containing chessboard data.
            chessboard_name (str): The name of the chessboard to retrieve coordinates for.

        Returns:
            Tuple[np.ndarray, int, int]: A tuple containing:
                                          - NumPy array of coordinates of the specified chessboard,
                                          - Number of rows in the chessboard array,
                                          - Number of columns in the chessboard array.
        """

        chessboard_coordinates = None
        num_rows = None
        num_columns = None
        for chessboard in data["chessboards"]:
            if chessboard["name"] == chessboard_name:
                chessboard_coordinates = chessboard["coordinates"]
                num_rows = chessboard["width"]
                num_columns = chessboard["height"]
                break

        num_dimensions = 3
        chessboard_array = np.zeros((num_rows, num_columns, num_dimensions))

        for idx, coord in enumerate(chessboard_coordinates):
            row = idx // num_columns
            column = idx % num_columns
            chessboard_array[row, column, 0] = coord["x"]
            chessboard_array[row, column, 1] = coord["y"]
            chessboard_array[row, column, 2] = coord["z"]

        return (np.float32(np.reshape(chessboard_array, (len(chessboard_coordinates), num_dimensions))),
                num_rows, num_columns)

    @staticmethod
    def draw_coord_system_object(img: np.ndarray, starting_point: np.ndarray, endpoint: np.ndarray) -> np.ndarray:
        """
        Draws a coordinate system object on the given image.

        Args:
            img (np.ndarray): The image on which to draw the coordinate system.
            starting_point (np.ndarray): The starting point of the coordinate system.
            endpoint (np.ndarray): The endpoints of the coordinate axes.

        Returns:
            np.ndarray: The image with the coordinate system drawn on it.
        """

        img = cv2.line(img, starting_point[0], tuple(endpoint[0].ravel()), (255, 0, 0), 5)
        img = cv2.line(img, starting_point[0], tuple(endpoint[1].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, starting_point[0], tuple(endpoint[2].ravel()), (0, 0, 255), 5)
        font = cv2.FONT_ITALIC
        cv2.putText(img, 'X', tuple(endpoint[0].ravel()), font, 5, (255, 0,), 2, cv2.LINE_AA)
        cv2.putText(img, 'Y', tuple(endpoint[1].ravel()), font, 5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, 'Z', tuple(endpoint[2].ravel()), font, 5, (0, 0, 255), 2, cv2.LINE_AA)

        return img

    @staticmethod
    def print_results(coords: Dict[str, Any], coord_sys: str) -> None:
        """
        Prints the coordinates of an object in a specified coordinate system.

        Args:
            coords (Dict[str, Any]): A dictionary containing the coordinates of the object.
            coord_sys (str): The coordinate system in which the coordinates are defined.

        Returns:
            None
        """

        rows = ['x', 'y', 'z']
        columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        coords = pd.DataFrame(coords, columns=columns, index=rows)
        logging.info(f"Object coordinates in the {coord_sys}")
        logging.info(coords)

    def plot_coords_object(self, coord_sys_type: str, rotation_vector: np.ndarray,
                           translation_vector: np.ndarray, starting_points: np.ndarray,
                           image: np.ndarray) -> None:
        """
        Plots coordinates of an object in a specified coordinate system on an image.

        Args:
            coord_sys_type (str): The type of coordinate system ('camera' or 'world').
            rotation_vector (np.ndarray): The rotation vector of the object.
            translation_vector (np.ndarray): The translation vector of the object.
            starting_points (np.ndarray): The starting points of the coordinate system.
            image (np.ndarray): The image on which to plot the coordinates.

        Returns:
            None
        """

        if coord_sys_type == 'camera':
            axis = np.float32([[157.48, 0, 0], [0, 204.47, 0], [0, 0, -25]]).reshape(-1, 3)
        elif coord_sys_type == 'world':
            axis = np.float32([[375, 0, 0], [0, 250, 0], [0, 0, -7]]).reshape(-1, 3)
        else:
            raise ValueError(f'Wrong coordinate system type given: {coord_sys_type}')

        axis_points, _ = cv2.projectPoints(axis,
                                           rotation_vector,
                                           translation_vector,
                                           self.undistorted_camera_mtx,
                                           distCoeffs=0)

        axis_points = np.squeeze(axis_points.astype(int))

        obj_dst = self.draw_coord_system_object(image, starting_points, axis_points)

        if coord_sys_type == 'camera':
            out_file_name = os.path.join(self.object_camera_cord_sys_path, "object_camera_cord_sys.jpg")
            cv2.imwrite(out_file_name, obj_dst)
        else:
            out_file_name = os.path.join(self.object_camera_world_sys_path, "object_camera_world_sys.jpg")
            cv2.imwrite(out_file_name, obj_dst)

    def object_pose_camera_coord_system(self) -> None:
        """
        Determines the object pose in the camera coordinate system.

        Returns:
            None
        """

        _, self.rotation_vector_object, self.translation_vector_object, _ = cv2.solvePnPRansac(
            objectPoints=self.motherboard_model_points,
            imagePoints=self.mounting_holes_data,
            cameraMatrix=self.undistorted_camera_mtx,
            distCoeffs=self.dist_coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        self.rotation_matrix_object, jacobin = cv2.Rodrigues(self.rotation_vector_object)
        self.xc = (
                dot(self.rotation_matrix_object, self.motherboard_model_points.T) + self.translation_vector_object
        )

        self.print_results(self.xc, "camera")

        starting_points = np.int32(self.mounting_holes_data[0:1])
        self.plot_coords_object("camera", self.rotation_vector_object, self.translation_vector_object,
                                starting_points, self.motherboard_image)

    def object_pose_world_coord_system(self) -> None:
        """
        Determines the object pose in the world coordinate system.

        Returns:
            None
        """

        gray = cv2.cvtColor(self.chessboard_image, cv2.COLOR_BGR2GRAY)
        return_corners, corners = cv2.findChessboardCorners(gray, (self.num_rows, self.num_cols), None)

        assert return_corners

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.001)
        chessboard_image_points = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        chessboard_image_points = np.squeeze(chessboard_image_points)

        _, self.rotation_vector_chessboard, self.translation_vector_chessboard, _ = \
            cv2.solvePnPRansac(self.chessboard_model_points,
                               chessboard_image_points,
                               self.undistorted_camera_mtx,
                               distCoeffs=self.dist_coefficients,
                               flags=cv2.SOLVEPNP_ITERATIVE)

        self.rotation_matrix_chessboard, jacobin = cv2.Rodrigues(self.rotation_vector_chessboard)
        self.rotation_inv_matrix_chessboard = np.linalg.inv(self.rotation_matrix_chessboard)

        term = self.xc-self.translation_vector_chessboard
        self.xw = dot(self.rotation_inv_matrix_chessboard, term)

        self.print_results(self.xw, "world")
        starting_point = chessboard_image_points[0].astype(int).reshape(1, 2)

        self.plot_coords_object("world", self.rotation_vector_chessboard,
                                self.translation_vector_chessboard, starting_point, self.chessboard_image)

    def main(self) -> None:
        """
        Executes the pose estimation of the motherboard.

        Returns:
             None
        """

        self.object_pose_camera_coord_system()
        self.object_pose_world_coord_system()


if __name__ == "__main__":
    try:
        pose_estimation = PoseEstimation()
        pose_estimation.main()
    except KeyboardInterrupt as kie:
        logging.error(f"{kie}")
