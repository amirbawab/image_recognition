#include <image2images/File.h>
#include <iostream>

std::shared_ptr<Image> File::loadImage() {
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
    for(int i=0; i < PIXLES; i++) {
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