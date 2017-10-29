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

void Image::cleanNoise() {
    // Binary threshold
    cv::threshold(*m_mat, *m_mat, 70, 255 /*white background*/, CV_THRESH_BINARY_INV);

    // Dilate objects to merge small parts
    cv::dilate(*m_mat, *m_mat, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2)));

    // Keep largest contours
    std::vector< std::vector<cv::Point> > contours;
    cv::findContours(*m_mat, contours, cv::RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // If there more than the 3 elements detected
    // then keep the largest 3 and delete the others
    if(contours.size() > NUM_OBJECTS) {

        // Store <index,area> in vector
        std::vector<std::pair<size_t, double>> pairs;
        for (size_t i=0; i<contours.size(); i++) {
            double area = cv::contourArea(contours[i], false);
            pairs.push_back(std::make_pair(i, area));
        }

        // Sort vector
        std::sort(pairs.begin(), pairs.end(),
                  [&](const std::pair<size_t, double>& firstElem, const std::pair<size_t , double >& secondElem) {
            return firstElem.second > secondElem.second;
        });

        // Delete everything after index 3
        for(size_t i=NUM_OBJECTS; i < pairs.size(); i++) {
            cv::drawContours(*m_mat, contours, (int) pairs[i].first, 0, CV_FILLED);
        }
    }
}

void Image::contour() {
    std::vector< std::vector<cv::Point> > contours;
    cv::findContours(*m_mat, contours, cv::RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // Loop on each object
    for (size_t i=0; i<contours.size(); i++) {

        // Store points
        std::vector<cv::Point> points;
        for (size_t j = 0; j < contours[i].size(); j++) {
            cv::Point p = contours[i][j];
            points.push_back(p);
        }

        // Draw rectangle
        if(points.size() > 0){
            cv::Rect brect = cv::boundingRect(cv::Mat(points).reshape(2));
            cv::rectangle(*m_mat, brect.tl() - cv::Point(2,2), brect.br()+cv::Point(2,2), cv::Scalar(220, 100, 200) /*color*/, 1 /*thickness*/, CV_AA);
        }
    }
}

void Image::wait() {
    cv::waitKey(0);
}