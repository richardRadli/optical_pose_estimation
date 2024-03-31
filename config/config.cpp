#include <iostream>
#include <string>
#include <map>
#include <filesystem>

namespace fs = std::filesystem;

class _Const{
    private:
        static std::string user;
        static std::map<std::string, std::string> root_mapping;
        static std::string DATA_ROOT;

public:
        static void initialize(){
            user = std::getenv("USER");
            if (root_mapping.find(user) == root_mapping.end()){
                throw std::invalid_argument("User name " + user + "not found");
            }

            for (const auto& [_, path]: dirs){

            }
        }
};
