//
// Created by rrb12 on 2024. 03. 31..
//

#ifndef OPTICAL_POSE_ESTIMATION_CPP_CONST_H
#define OPTICAL_POSE_ESTIMATION_CPP_CONST_H

#include <string>
#include <map>

class Const {
public:
    static void initialize();
    static void create_directories(const std::map<std::string, std::string>& dirs, const std::string& root_type);

    static std::string user;
    static std::map<std::string, std::map<std::string, std::string>> root_mapping;
    static std::string DATA_ROOT;
};

class Images : public Const {
public:
    Images(); // Constructor declaration

    static std::string get_data_path(const std::string& key);

private:
    static std::map<std::string, std::string> dir_images;
};

class Data : public Const {
public:
    Data(); // Constructor declaration

    static std::string get_data_path(const std::string& key);

private:
    static std::map<std::string, std::string> dir_data;
};

#endif //OPTICAL_POSE_ESTIMATION_CPP_CONST_H
