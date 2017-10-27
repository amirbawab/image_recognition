cmake_minimum_required(VERSION 3.5)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include thirdparty libraries header files


# Compile with -std=c++11
add_compile_options(-std=c++11)

# Disable IPP optimization
#option(WITH_IPP OFF)

# Add sub-projects
#add_subdirectory(thirdparty/opencv)
add_subdirectory(demo)


