cmake_minimum_required(VERSION 3.5)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Compile with -std=c++11
add_compile_options(-std=c++11)

# Include header files
include_directories(neuralnetwork/include)

# Add sub-projects
add_subdirectory(demo)
add_subdirectory(neuralnetwork)
add_subdirectory(image2images)
#add_subdirectory(tesseract)


