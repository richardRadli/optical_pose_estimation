//
// Created by rrb12 on 2024. 03. 31..
//

#ifndef OPTICAL_POSE_ESTIMATION_CPP_CONFIG_SELECTOR_H
#define OPTICAL_POSE_ESTIMATION_CPP_CONFIG_SELECTOR_H

#include <map>
#include <string>

std::map<std::string, std::string> camera_config();
std::map<std::string, std::string> feature_matching_config();
std::map<std::string, std::string> pose_estimation_config();

#endif //OPTICAL_POSE_ESTIMATION_CPP_CONFIG_SELECTOR_H
