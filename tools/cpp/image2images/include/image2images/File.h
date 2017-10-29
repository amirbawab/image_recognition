#pragma once

#include <fstream>
#include <memory>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define ROWS 64
#define COLS 64
#define PIXLES ROWS * COLS

class File {
private:
    std::ifstream m_input;
public:
    File(std::string fileName) : m_input(fileName) {}

    /**
     * Load a matrix
     * @return matrix pointer
     */
    std::shared_ptr<cv::Mat> loadMat();

    /**
     * Skip mat in file
     */
    void skipMat();
};