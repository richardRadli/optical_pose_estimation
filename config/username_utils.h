//
// Created by rrb12 on 2024. 03. 31..
//

#ifndef OPTICAL_POSE_ESTIMATION_CPP_USERNAME_UTILS_H
#define OPTICAL_POSE_ESTIMATION_CPP_USERNAME_UTILS_H

#include <cstdlib>
#include <string>
#include <iostream>

inline std::string get_username() {
#ifdef _WIN32
    const char* env_var_name = "USERNAME";
#else
    const char* env_var_name = "USER";
#endif
    char* username = std::getenv(env_var_name);
    if (username) {
        return std::string(username);
    } else {
        std::cerr << "Unable to retrieve username from environment variable " << env_var_name << std::endl;
        return "default_username"; // Provide a default username
    }
}


#endif //OPTICAL_POSE_ESTIMATION_CPP_USERNAME_UTILS_H
