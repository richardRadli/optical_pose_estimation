#include <iostream>
#include <string>
#include <map>
#include <filesystem>

#include "const.h"
#include "username_utils.h"

namespace fs = std::filesystem;


// ---------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------- C O N S T -----------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void Const::initialize() {
    user = get_username();
    if (root_mapping.find(user) == root_mapping.end()) {
        throw std::invalid_argument("User name " + user + " not found in root_mapping.");
    }
    DATA_ROOT = root_mapping[user]["DATA_ROOT"];
}

void Const::create_directories(const std::map<std::string, std::string>& dirs, const std::string& root_type) {
    if (root_type != "DATA_ROOT") {
        throw std::invalid_argument("Root type " + root_type + " not found.");
    }
    for (const auto& [_, path] : dirs) {
        fs::path dir_path = DATA_ROOT + "/" + path;
        if (!fs::exists(dir_path)) {
            fs::create_directories(dir_path);
            std::cout << "Directory " << dir_path << " has been created" << std::endl;
        }
    }
}


std::string Const::user;
std::map<std::string, std::map<std::string, std::string>> Const::root_mapping = {
        {"rrb12", {
                {"DATA_ROOT", "C:/Users/rrb12/Documents/project/storage_cpp"}
        }}
};
std::string Const::DATA_ROOT;


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------- I M A G E S ----------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
Images::Images() {
    initialize();
    create_directories(dir_images, "DATA_ROOT");
}

std::string Images::get_data_path(const std::string& key) {
    return DATA_ROOT + "/" + dir_images[key];
}

std::map<std::string, std::string> Images::dir_images = {
        {"calibration_images", "images/camera/calibration_images"},
        {"chessboard_images", "images/camera/chessboard_images"},
        {"motherboard_images", "images/camera/motherboard_images"},
        {"undistorted_calibration_images", "images/objects/calibration_images"},
        {"undistorted_chessboard_images", "images/objects/chessboard_images"},
        {"undistorted_motherboard_images", "images/objects/motherboard_images"},
        {"roi_images", "images/pose_estimation/roi_images"},
        {"pairing_images", "images/pose_estimation/pairing_images"},
        {"object_camera_cord_sys", "images/pose_estimation/object_camera_cord_sys"},
        {"object_camera_world_sys", "images/pose_estimation/object_camera_world_sys"}
};


// ---------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------ D A T A ------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
Data::Data() {
    initialize();
    create_directories(dir_data, "ROOT_DIR");
}

std::string Data::get_data_path(const std::string& key) {
    return DATA_ROOT + "/" + dir_data[key];
}

std::map<std::string, std::string> Data::dir_data = {
        {"camera_matrix", "data/camera/camera_matrix"},
        {"camera_settings", "data/camera/camera_matrix"},
        {"model_coordinates", "data/camera/camera_settings"},
        {"mounting_hole_center_points", "data/pose_estimation/mounting_hole_center_points"}
};
