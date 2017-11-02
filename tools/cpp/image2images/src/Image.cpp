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
    winName << "Image " << m_name << " # " << m_id;
    cv::namedWindow(winName.str(), cv::WINDOW_AUTOSIZE);
    cv::imshow(winName.str(), *m_mat);
}

void Image::_reduceColors(int K) {
    int n = m_mat->rows * m_mat->cols;
    cv::Mat data = m_mat->reshape(1, n);
    data.convertTo(data, CV_32F);

    std::vector<int> labels;
    cv::Mat1f colors;
    cv::kmeans(data, K, labels
            , cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001)
            , 5, cv::KMEANS_PP_CENTERS, colors);

    for (int i = 0; i < n; ++i) {
        data.at<float>(i, 0) = colors(labels[i], 0);
    }

    cv::Mat reduced = data.reshape(1, m_mat->rows);
    reduced.convertTo(reduced, CV_8U);
    reduced.copyTo(*m_mat);
}

void Image::_binarize(std::shared_ptr<Image> binImage, int threshold) {
    // Create a new matrix
    binImage->m_mat = std::make_shared<cv::Mat>(m_mat->rows, m_mat->cols, m_mat->type());
    m_mat->copyTo(*binImage->m_mat);

    // Apply binary threshold
    cv::threshold(*binImage->getMat(), *binImage->getMat(), threshold, 255, CV_THRESH_BINARY_INV);

    // Dilate objects to merge small parts
    cv::dilate(*binImage->getMat(), *binImage->getMat(), cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(m_mat->rows/64 + 2, m_mat->cols/64 + 2)));

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
    std::shared_ptr<Image> binImage = clone();

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

    std::shared_ptr<Image> alignedImage = _buildImage(ALIGN_ROWS, ALIGN_COLS, indices);

    // FIXME Right now if aligning the images is not possible, then we return a solid black image
    if(!alignedImage) {
        alignedImage = clone();
        alignedImage->m_mat = std::make_shared<cv::Mat>(ALIGN_ROWS, ALIGN_COLS, m_mat->type());
        alignedImage->m_mat->setTo(cv::Scalar(0));
    }
    return alignedImage;
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
    std::shared_ptr<Image> image = clone();
    image->m_mat = std::make_shared<cv::Mat>(rows, cols, m_mat->type());
    image->getMat()->setTo(cv::Scalar(0));

    // Start populating the new matrix
    int leftOffset = 0;
    int topOffset = padding;
    for(size_t i=0; i < indices.size(); i++) {

        // Update left offset
        leftOffset += padding;

        // Get object
        cv::Rect brect = cv::boundingRect(cv::Mat(m_contours[indices[i]]).reshape(2));
        cv::Mat elementMat = ((*m_mat)(cv::Rect(brect.tl(), brect.br())));

        // Verify that the new image fits the matrix
        // This maximizes the number of potential well
        // formed images
        if(leftOffset + brect.width >= image->getMat()->cols || topOffset + brect.height >= image->getMat()->rows) {
            return nullptr;
        }

        // Draw element on new matrix
        elementMat(cv::Rect(0,0,brect.width, brect.height)).copyTo(
                (*image->getMat())(cv::Rect(leftOffset, padding, brect.width, brect.height)));

        // Update left offset
        leftOffset += brect.width + padding;
    }
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

std::shared_ptr<Image> Image::rotate(int angle) {
    std::shared_ptr<Image> rotImage = clone();
    cv::Point2f src_center(rotImage->getMat()->cols/2.0F, rotImage->getMat()->rows/2.0F);
    cv::Mat rotMat = getRotationMatrix2D(src_center, angle, 1.0);
    cv::warpAffine(*rotImage->getMat(), *rotImage->getMat(), rotMat, rotImage->getMat()->size());
    return rotImage;
}

std::vector<std::shared_ptr<Image>> Image::mnist() {
    // Prepare mnist vector
    std::vector<std::shared_ptr<Image>> mnistVector;
    mnistVector.push_back(shared_from_this());

    // Invert colors
    cv::bitwise_not (*m_mat, *m_mat);

    // Copy matrix to a larger one
    cv::copyMakeBorder(*m_mat, *m_mat, 11,11,11,11, cv::BORDER_CONSTANT, cv::Scalar(0)); // 28x28 -> 50x50

    // Rotate images
    std::shared_ptr<Image> LRImage = rotate(45);
    std::shared_ptr<Image> RRImage = rotate(-45);

    // Add new images
    mnistVector.push_back(LRImage);
    mnistVector.push_back(RRImage);
    return mnistVector;
}

std::shared_ptr<Image> Image::clone() {
    std::shared_ptr<cv::Mat> mat = std::make_shared<cv::Mat>(m_mat->rows, m_mat->cols, m_mat->type());
    m_mat->copyTo(*mat);
    std::shared_ptr<Image> image = std::make_shared<Image>(mat, m_value);
    return image;
}

std::shared_ptr<Image> Image::scale(double val) {
    std::shared_ptr<Image> image = clone();
    cv::resize(*image->getMat(), *image->getMat(), cv::Size((int)(image->getMat()->rows*val), (int)(image->getMat()->cols*val)));
    return image;
}

std::string Image::recognize() {
    return "";
}