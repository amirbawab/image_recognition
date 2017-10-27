project(demo)

# Add exectuable
add_executable(${PROJECT_NAME} src/demo.cpp)

# Add libraries
target_link_libraries(${PROJECT_NAME}
        opencv_core
        opencv_imgproc
        opencv_highgui
        opencv_ml
        opencv_video
        opencv_features2d
        opencv_calib3d
        opencv_objdetect
        opencv_imgcodecs
        opencv_flann)
