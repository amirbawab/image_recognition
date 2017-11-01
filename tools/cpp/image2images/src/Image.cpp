#include <image2images/Image.h>
#include <iostream>

#define NUM_OBJECTS 3
#define MIN_OBJECT_AREA 30
#define ALIGN_ROWS 70
#define ALIGN_COLS 100
#define SPLIT_ROWS 50
#define SPLIT_COLS 50

// Init unique id val
long Image::m_uniq_id = 1;

void Image::display() {
    std::stringstream winName;
    winName << "Image " << m_id;
    cv::namedWindow(winName.str(), cv::WINDOW_AUTOSIZE);
    cv::imshow(winName.str(), *m_mat);
}

void Image::_binarize(std::shared_ptr<Image> binImage, int threshold) {
    // Create a new matrix
    binImage->m_mat = std::make_shared<cv::Mat>(m_mat->rows, m_mat->cols, m_mat->type());

    // Apply binary threshold
    cv::threshold(*m_mat, *binImage->getMat(), threshold, 255, CV_THRESH_BINARY_INV);

    // Dilate objects to merge small parts
    cv::dilate(*binImage->getMat(), *binImage->getMat(), cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2)));

    // Detect elements
    binImage->detectElements();
}

double Image::_averagePixelVal() {
    double sum = 0;
    for(int row=0; row < m_mat->rows; row++) {
        for(int col=0; col < m_mat->cols; col++) {
            sum += (int) m_mat->at<uchar>(row, col);
        }
    }
    return sum / (m_mat->rows * m_mat->cols);
}

std::shared_ptr<Image> Image::binarize(int threshold) {
    std::shared_ptr<Image> binImage = std::make_shared<Image>();

    // Average pixel threshold
    const int AVG_THERSHOLD = 50;

    // If threshold is 0, then try to get the best
    // binary image
    if(threshold == 0) {
        int startThreshold = (int)_averagePixelVal();
        do{
            _binarize(binImage, startThreshold);
            startThreshold -= 5;
        } while(startThreshold > 0 && (binImage->m_contours.size() > NUM_OBJECTS || binImage->_averagePixelVal() > AVG_THERSHOLD));
    } else {
        _binarize(binImage, threshold);
    }
    binImage->m_value = m_value;
    return binImage;
}

void Image::detectElements() {
    // Find contours
    cv::findContours(*m_mat, m_contours, cv::RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
}

std::shared_ptr<Image> Image::align() {
    std::vector<int> indices;
    for(int i=0; i < m_contours.size(); i++) {
        indices.push_back(i);
    }
    return _buildImage(ALIGN_ROWS, ALIGN_COLS, indices);
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
        size_t index = 0;
        for(size_t i=0; i < NUM_OBJECTS; i++) {
            if(pairs[i].second >= MIN_OBJECT_AREA) {
                keepContours.push_back(m_contours[pairs[i].first]);
                index++;
            }
        }

        // Delete everything after index
        for(size_t i=index; i < pairs.size(); i++) {
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

std::shared_ptr<Image> Image::_buildImage(int rows, int cols, const std::vector<int> &indices) {

    // Compute new width and height
    int padding = 3;

    // Construct and initialize a new mat
    std::shared_ptr<cv::Mat> mat = std::make_shared<cv::Mat>(rows, cols, m_mat->type());
    mat->setTo(cv::Scalar(0));

    // Construct image
    std::shared_ptr<Image> image = std::make_shared<Image>(mat);

    // Start populating the new matrix
    int leftOffset = 0;
    int topOffset = padding;
    for(size_t i=0; i < indices.size(); i++) {

        // Update left offset
        leftOffset += padding;

        // Get object
        cv::Rect brect = cv::boundingRect(cv::Mat(m_contours[indices[i]]).reshape(2));
        cv::Mat elementMat = ((*m_mat)(cv::Rect(brect.tl(), brect.br())));

        // Deskew element
        // FIXME Experimental
//        _deskew(elementMat);

        // Verify that the new image fits the matrix
        // This maximizes the number of potential well
        // formed images
        if(leftOffset + brect.width >= mat->cols || topOffset + brect.height >= mat->rows) {
            return nullptr;
        }

        // Draw element on new matrix
        elementMat(cv::Rect(0,0,brect.width, brect.height)).copyTo(
                (*mat)(cv::Rect(leftOffset, padding, brect.width, brect.height)));

        // Update left offset
        leftOffset += brect.width + padding;
    }

    // Update value
    image->m_value = m_value;
    return image;
}

void Image::_permutation(std::vector<std::shared_ptr<Image>> &outputImages, std::vector<int> &indices) {
    if(indices.size() == m_contours.size()) {
        std::shared_ptr<Image> newImage = _buildImage(ALIGN_ROWS, ALIGN_COLS, indices);
        if(newImage) {
            outputImages.push_back(newImage);
        }
    } else {
        for(int i=0; i < m_contours.size(); i++) {
            if(std::find(indices.begin(), indices.end(), i) == indices.end()) {
                indices.push_back(i);
                _permutation(outputImages, indices);
                indices.pop_back();
            }
        }
    }
}

std::vector<std::shared_ptr<Image>> Image::permutation() {
    std::vector<std::shared_ptr<Image>> perImages;
    std::vector<int> indices;
    _permutation(perImages, indices);
    return perImages;
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

std::vector<std::shared_ptr<Image>> Image::split() {
    std::vector<std::shared_ptr<Image>> splitImages;
    for(int i=0; i < m_contours.size(); i++) {
        std::vector<int> indices = {i};
        std::shared_ptr<Image> elementImage = _buildImage(SPLIT_ROWS, SPLIT_COLS, indices);
        if(elementImage) {
            splitImages.push_back(elementImage);
        }
    }
    return splitImages;
}

void Image::wait() {
    cv::waitKey(0);
}

void Image::rotate(int angle) {
    cv::Point2f center(m_mat->cols/2.0f, m_mat->rows/2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Rect bbox = cv::RotatedRect(center,m_mat->size(), angle).boundingRect();
    rot.at<double>(0,2) += bbox.width/2.0 - center.x;
    rot.at<double>(1,2) += bbox.height/2.0 - center.y;
    cv::warpAffine(*m_mat, *m_mat, rot, bbox.size());
}

std::vector<std::shared_ptr<Image>> Image::mnist() {
    // Prepare mnist vector
    std::vector<std::shared_ptr<Image>> mnistVector;
    mnistVector.push_back(shared_from_this());

    // Prepare the new images
    std::shared_ptr<cv::Mat> leftRotMat = std::make_shared<cv::Mat>(m_mat->rows, m_mat->cols, m_mat->type());
    std::shared_ptr<cv::Mat> rightRotMat = std::make_shared<cv::Mat>(m_mat->rows, m_mat->cols, m_mat->type());
    m_mat->copyTo(*leftRotMat);
    m_mat->copyTo(*rightRotMat);
    mnistVector.push_back(std::make_shared<Image>(leftRotMat, m_value));
    mnistVector.push_back(std::make_shared<Image>(rightRotMat, m_value));

    // Manipulate images
    cv::Point2f src_center(mnistVector[0]->getMat()->cols/2.0F, mnistVector[0]->getMat()->rows/2.0F);
    cv::Mat LRot = getRotationMatrix2D(src_center, -45, 1.0);
    cv::Mat RRot = getRotationMatrix2D(src_center, 45, 1.0);

    // Apply rotation
    warpAffine(*mnistVector[0]->getMat(), *mnistVector[0]->getMat(), LRot, mnistVector[0]->getMat()->size());
    warpAffine(*mnistVector[1]->getMat(), *mnistVector[1]->getMat(), RRot, mnistVector[1]->getMat()->size());

    return mnistVector;
}