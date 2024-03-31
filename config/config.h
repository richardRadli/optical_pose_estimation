//
// Created by rrb12 on 2024. 03. 31..
//

#ifndef OPTICAL_POSE_ESTIMATION_CPP_CONFIG_H
#define OPTICAL_POSE_ESTIMATION_CPP_CONFIG_H

#include <string>


class CameraAndCalibrationConfig {
public:
    static const int HEIGHT;
    static const int WIDTH;
    static const int CAM_ID;
    static const int CHS_COL;
    static const int CHS_ROW;
    static const int SQUARE_SIZE;
    static const float ERROR_THRESHOLD;
    static const int SIZE_COEFFICIENTS;
    static const bool UNDISTORT;
    static const int CPU_THREADS;
    static const std::string OPERATION;
};


class FeatureMatchingConfig{
public:
    static const int SIZE_COEFFICIENT;
    static const int KERNEL_SIZE;
    static const int DP;
    static const int MINDIST;
    static const int PARAM1;
    static const int PARAM2;
    static const int MIN_RADIUS;
    static const int MAX_RADIUS;
    static const int NFEATURES;
    static const int NOCTAVE_LAYERS;
    static const float CONTRAST_THRESHOLD;
    static const int EDGE_THRESHOLD;
    static const float SIGMA;
    static const int MIN_MATCH_COUNT;
    static const float THRESHOLD;
};


class PoseEstimationConfig{
public:
    static const std::string CHESSBOARD_TYPE;
    static const std::string MOTHERBOARD_TYPE;
};

#endif //OPTICAL_POSE_ESTIMATION_CPP_CONFIG_H
