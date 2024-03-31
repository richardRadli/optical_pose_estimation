//
// Created by rrb12 on 2024. 03. 31..
//

#include "config.h"

const int CameraAndCalibrationConfig::SIZE_COEFF = 3;
const int CameraAndCalibrationConfig::HEIGHT = 2448;
const int CameraAndCalibrationConfig::WIDTH = 3264;
const int CameraAndCalibrationConfig::CAM_ID = 0;
const int CameraAndCalibrationConfig::CHS_COL = 15;
const int CameraAndCalibrationConfig::CHS_ROW = 10;
const int CameraAndCalibrationConfig::SQUARE_SIZE = 25;
const float CameraAndCalibrationConfig::ERROR_THRESHOLD = 0.2;
const int CameraAndCalibrationConfig::SIZE_COEFFICIENTS = 3;
const bool CameraAndCalibrationConfig::UNDISTORT = false;
const int CameraAndCalibrationConfig::CPU_THREADS = 6;
const std::string CameraAndCalibrationConfig::OPERATION = "calibration_images";

const int FeatureMatchingConfig::PARAM1 = 50;
const int FeatureMatchingConfig::PARAM2 = 30;
const int FeatureMatchingConfig::MIN_RADIUS = 15;
const int FeatureMatchingConfig::MAX_RADIUS = 40;
const int FeatureMatchingConfig::NFEATURES = 0;
const int FeatureMatchingConfig::NOCTAVE_LAYERS = 5;
const float FeatureMatchingConfig::CONTRAST_THRESHOLD = 0.05;
const int FeatureMatchingConfig::EDGE_THRESHOLD = 40;
const float FeatureMatchingConfig::SIGMA = 1.0;
const int FeatureMatchingConfig::MIN_MATCH_COUNT = 25;
const float FeatureMatchingConfig::THRESHOLD = 0.6;

const std::string PoseEstimationConfig::CHESSBOARD_TYPE = "A3";
const std::string PoseEstimationConfig::MOTHERBOARD_TYPE = "ATX";
