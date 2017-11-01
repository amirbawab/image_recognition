#pragma once

#include <fstream>
#include <memory>
#include <image2images/Image.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class File {
private:
    int m_imgRows;
    int m_imgCols;
    std::ifstream m_input;
    std::ifstream m_label;
public:
    File(int rows, int cols) : m_imgRows(rows), m_imgCols(cols) {}

    /**
     * Load a matrix
     * @return matrix pointer
     */
    std::shared_ptr<Image> loadImage();

    /**
     * Skip mat in file
     */
    void skipMat();

    /**
     * Set label file
     * @param label
     * @return true if file is open
     */
    bool setLabelFile(std::string label);

    /**
     * Set input file
     * @param input
     * @return true if file is open
     */
    bool setInputFile(std::string input);
};