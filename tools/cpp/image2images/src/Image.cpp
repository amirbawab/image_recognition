#include <image2images/Image.h>
#include <iostream>

#define NUM_OBJECTS 3

// Init unique id val
unsigned int Image::m_uniq_id = 1;

void Image::display() {
    std::stringstream winName;
    winName << "Image " << m_id;
    cv::namedWindow(winName.str(), cv::WINDOW_AUTOSIZE);
    cv::imshow(winName.str(), *m_mat);
}

void Image::extract() {
    cv::findContours(*m_mat, m_contours, cv::RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
}

void Image::binarize() {
    // Apply binary threshold
    cv::threshold(*m_mat, *m_mat, 70, 255 /*white background*/, CV_THRESH_BINARY_INV);

    // Dilate objects to merge small parts
    cv::dilate(*m_mat, *m_mat, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2)));
}

void Image::cleanNoise() {

    // If there more than the 3 elements detected
    // then keep the largest 3 and delete the others
    if(m_contours.size() > NUM_OBJECTS) {

        // Store <index,area> in vector
        std::vector<std::pair<size_t, double>> pairs;
        for (size_t i=0; i< m_contours.size(); i++) {
            double area = cv::contourArea(m_contours[i], false);
            pairs.push_back(std::make_pair(i, area));
        }

        // Sort vector
        std::sort(pairs.begin(), pairs.end(),
                  [&](const std::pair<size_t, double>& firstElem, const std::pair<size_t , double >& secondElem) {
            return firstElem.second > secondElem.second;
        });

        // Store new contour
        std::vector< std::vector<cv::Point>> keepContours;
        for(size_t i=0; i < NUM_OBJECTS; i++) {
            keepContours.push_back(m_contours[pairs[i].first]);
        }

        // Delete everything after index 3
        for(size_t i=NUM_OBJECTS; i < pairs.size(); i++) {
            cv::drawContours(*m_mat, m_contours, (int) pairs[i].first, 0, CV_FILLED);
        }

        // Update contour member
        m_contours = keepContours;
    }
}

void Image::contour() {
    // Loop on each object
    for (size_t i=0; i<m_contours.size(); i++) {

        // Store points
        std::vector<cv::Point> points;
        for (size_t j = 0; j < m_contours[i].size(); j++) {
            cv::Point p = m_contours[i][j];
            points.push_back(p);
        }

        // Draw rectangle
        if(points.size() > 0){
            cv::Rect brect = cv::boundingRect(cv::Mat(points).reshape(2));
            cv::rectangle(*m_mat, brect.tl() - cv::Point(2,2), brect.br()+cv::Point(2,2), cv::Scalar(220, 100, 200) /*color*/, 1 /*thickness*/, CV_AA);
        }
    }
}

std::shared_ptr<cv::Mat> Image::recreate() {

    // Compute new width and height
    int padding = 5;
    int rows = 0;
    int cols = padding;
    for(size_t i=0; i < m_contours.size(); i++) {
        cv::Rect brect = cv::boundingRect(cv::Mat(m_contours[i]).reshape(2));
        cols += brect.width  + padding;
        rows = std::max(rows, brect.height);
    }
    rows += padding*2;

    // Construct and initialize a new mat
    std::shared_ptr<cv::Mat> mat = std::make_shared<cv::Mat>(rows, cols, m_mat->type());
    mat->setTo(cv::Scalar(0));

    // Start populating the new matrix
    int leftOffset = padding;
    for(size_t i=0; i < m_contours.size(); i++) {

        // Get object
        cv::Rect brect = cv::boundingRect(cv::Mat(m_contours[i]).reshape(2));
        cv::Mat elementMat = ((*m_mat)(cv::Rect(brect.tl(), brect.br())));

        // Deskew element
        // FIXME Experimental
//        _deskew(elementMat);

        // Draw element on new matrix
        elementMat(cv::Rect(0,0,brect.width, brect.height)).copyTo(
                (*mat)(cv::Rect(leftOffset, padding, brect.width, brect.height)));

        // Update left offset
        leftOffset += brect.width+padding;
    }
    return mat;
}

void Image::_deskew(cv::Mat &mat){
    int SZ = 20;
    int affineFlags = cv::WARP_INVERSE_MAP | cv::INTER_LINEAR;
    cv::Moments m = cv::moments(*m_mat);
    if(abs(m.mu02) < 1e-2) {
        return;
    }
    double skew = m.mu11/m.mu02;
    cv::Mat_<float> warpMat(2,3);
    warpMat << 1, skew, -0.5*SZ*skew, 0, 1, 0;
    warpAffine(mat, mat, warpMat, m_mat->size(), affineFlags);
}

void Image::wait() {
    cv::waitKey(0);
}