//
// Created by rrb12 on 2024. 03. 31..
//

#include <iostream>
#include "config_selector.h"
#include "const.h"

std::map<std::string, std::string> camera_config() {
    std::map<std::string, std::string> cam_config;

    cam_config["calibration_images"] = Images::get_data_path("calibration_images");
    cam_config["chessboard_images"] = Images::get_data_path("chessboard_images");
    cam_config["motherboard_images"] = Images::get_data_path("motherboard_images");
    cam_config["undistorted_calibration_images"] = Images::get_data_path("undistorted_calibration_images");
    cam_config["undistorted_chessboard_images"] = Images::get_data_path("undistorted_chessboard_images");
    cam_config["undistorted_motherboard_images"] = Images::get_data_path("undistorted_motherboard_images");
    cam_config["camera_matrix"] = Data::get_data_path("camera_matrix");
    cam_config["camera_settings"] = Data::get_data_path("camera_settings");

    return cam_config;
}

std::map<std::string, std::string> feature_matching_config() {
    std::map<std::string, std::string> feature_matching;

    feature_matching["roi_images"] = Images::get_data_path("roi_images");
    feature_matching["pairing_images"] = Images::get_data_path("pairing_images");
    feature_matching["mounting_hole_center_points"] = Data::get_data_path("mounting_hole_center_points");

    return feature_matching;
}


std::map<std::string, std::string> pose_estimation_config() {
    std::map<std::string, std::string> pose_estimation;

    pose_estimation["chessboard_model_points"] = Data::get_data_path("chessboard_model_points");
    pose_estimation["motherboard_model_points"] = Data::get_data_path("motherboard_model_points");
    pose_estimation["object_camera_cord_sys"] = Data::get_data_path("object_camera_cord_sys");
    pose_estimation["object_camera_world_sys"] = Data::get_data_path("object_camera_world_sys");

    return pose_estimation;
}

