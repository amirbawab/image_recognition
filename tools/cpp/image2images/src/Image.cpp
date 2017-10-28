#include <image2images/Image.h>

// Init unique id val
unsigned int Image::m_uniq_id = 1;

void Image::display() {
    std::stringstream winName;
    winName << "Image " << m_id;
    cv::namedWindow(winName.str(), cv::WINDOW_AUTOSIZE);
    cv::imshow(winName.str(), *m_mat);
}

void Image::cleanNoise() {
    cv::threshold(*m_mat, *m_mat, 70, 255 /*white background*/, CV_THRESH_BINARY_INV);
}

void Image::contour() {
    std::vector< std::vector<cv::Point> > contours;
    cv::findContours(*m_mat, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

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
            cv::rectangle(*m_mat, brect.tl(), brect.br(), cv::Scalar(220, 100, 200) /*color*/, 1 /*thickness*/, CV_AA);
        }
    }
}

void Image::wait() {
    cv::waitKey(0);
}