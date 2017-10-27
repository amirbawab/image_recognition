project(demo)

# Add exectuable
add_executable(${PROJECT_NAME} src/demo.cpp)

# Add opencv
find_package( OpenCV REQUIRED )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
