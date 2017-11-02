#include <image2images/File.h>
#include <iostream>

std::shared_ptr<Image> File::loadImage() {
    std::shared_ptr<cv::Mat> out = std::make_shared<cv::Mat>(m_imgRows, m_imgCols, CV_8UC1);
    int row = 0;
    int col = 0;
    int pixels = m_imgRows * m_imgCols;
    for(int i=0; i < pixels; i++) {

        // Load input from file
        double val;
        m_input >> val;

        // Update pixel
        out->at<uchar>(row, col) = (uchar)(255 - val);
        if(++col == m_imgCols) {
            row++;
            col = 0;
        }
    }

    std::shared_ptr<Image> image;
    if(m_label.is_open()) {
        if(m_label.peek() == EOF) {
            std::cerr << ">> WARNING: Label file reached EOF, image was not tagged" << std::endl;
            image = std::make_shared<Image>(out);
        } else {
            int val;
            m_label >> val;
            image = std::make_shared<Image>(out, val);
        }
    } else {
        image = std::make_shared<Image>(out);
    }
    return image;
}

void File::skipMat() {
    double ignore;
    int pixels = m_imgRows * m_imgCols;
    for(int i=0; i < pixels; i++) {
        m_input >> ignore;
    }

    // Ignore the label as well
    if(m_label.is_open() && m_label.peek() != EOF) {
        int val;
        m_label >> val;
    }
}

bool File::setLabelFile(std::string label) {
    m_label.open(label);
    return m_label.is_open();
}

bool File::setInputFile(std::string input) {
    m_input.open(input);
    return m_input.is_open();
}