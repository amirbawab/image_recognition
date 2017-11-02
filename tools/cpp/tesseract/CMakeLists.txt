project(ocr)

# Add exectuable
add_executable(${PROJECT_NAME} src/tesseract.cpp)

# Link library
target_link_libraries (${PROJECT_NAME} tesseract lept)
