#include <iostream>

#include "config/const.h"
#include "config/config.h"


int main() {
    try {
        // Initialize Const class
        Const::initialize();

        // Create an instance of the Images class
        Images images;

        // Create an instance of the Data class
        Data data;

    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}