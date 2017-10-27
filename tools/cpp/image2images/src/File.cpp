#include <image2images/File.h>

/**
 * Read a matrix from input
 * @return matrix
 */
std::shared_ptr<cv::Mat> File::loadMat() {
    std::shared_ptr<cv::Mat> out = std::make_shared<cv::Mat>(ROWS, COLS, CV_8UC1);
    int row = 0;
    int col = 0;
    for(int i=0; i < PIXLES; i++) {

        // Load input from file
        double val;
        m_input >> val;

        // Update pixel
        out->at<uchar>(row, col) = (uchar)(255 - val);
        if(++col == COLS) {
            row++;
            col = 0;
        }
    }
    return out;
}
