#include <image2images/Image.h>
#include <iostream>

#define NUM_OBJECTS 3

// Init unique id val
long Image::m_uniq_id = 1;

void Image::display() {
    std::stringstream winName;
    winName << "Image " << m_name << " # " << m_id;
    cv::namedWindow(winName.str(), cv::WINDOW_AUTOSIZE);
    cv::imshow(winName.str(), *m_mat);
}

void Image::_binarize(std::shared_ptr<Image> binImage, int threshold) {
    // Create a new matrix
    binImage->m_mat = _cloneMat();

    // Apply binary threshold
    cv::threshold(*binImage->getMat(), *binImage->getMat(), threshold, 255, CV_THRESH_BINARY);

    // Dilate objects to merge small parts
    cv::dilate(*binImage->getMat(), *binImage->getMat(), cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(m_mat->rows/64 + 2, m_mat->cols/64 + 2)));
}

double Image::_averagePixelVal() {
    double sum = 0;
    for(int row=0; row < getSide(); row++) {
        for(int col=0; col < getSide(); col++) {
            sum += (int) m_mat->at<uchar>(row, col);
        }
    }
    return sum / (getSide()* getSide());
}

std::shared_ptr<Image> Image::binarize(int threshold) {
    std::shared_ptr<Image> binImage = _cloneImage();

    // Average pixel threshold (Based on experiments)
    const int AVG_THERSHOLD = 50;

    // If threshold is 0, then try to get the best
    // binary image
    if(threshold == 0) {
        int startThreshold = (int)_averagePixelVal();
        do{
            _binarize(binImage, startThreshold);
            startThreshold += 5;
        } while(startThreshold < 255 && (binImage->_charsControus().size() > NUM_OBJECTS
                                       || binImage->_averagePixelVal() > AVG_THERSHOLD));
    } else {
        _binarize(binImage, threshold);
    }
    return binImage;
}

std::shared_ptr<Image> Image::align() {
    std::vector<int> indices;
    std::vector<std::vector<cv::Point>> charsContour = _charsControus();
    for(int i=0; i < charsContour.size(); i++) {
        indices.push_back(i);
    }
    return _align(indices, charsContour);
}

std::shared_ptr<Image> Image::cleanNoise() {

    // Keep only the real characters
    std::vector<std::vector<cv::Point>> charsContour = _charsControus();
    std::shared_ptr<Image> cleanImage = _cloneImage();

    // If no need to clean, return a clone
    if(charsContour.size() <= NUM_OBJECTS) {
        return cleanImage;
    }

    // Store <index,area> in vector
    std::vector<std::pair<size_t, double>> pairs;
    for (size_t i=0; i< charsContour.size(); i++) {
        double area = cv::contourArea(charsContour[i], false);
        pairs.push_back(std::make_pair(i, area));
    }

    // Sort vector
    std::sort(pairs.begin(), pairs.end(),
              [&](const std::pair<size_t, double>& firstElem, const std::pair<size_t , double >& secondElem) {
        return firstElem.second > secondElem.second;
    });

    // Erase everything noise
    for(size_t i=NUM_OBJECTS; i < pairs.size(); i++) {
        cv::drawContours(*cleanImage->getMat(), charsContour, (int) pairs[i].first, 0, CV_FILLED);
    }
    return cleanImage;
}

std::shared_ptr<Image> Image::drawContour() {

    // Clone current image
    std::shared_ptr<Image> contourImage = _cloneImage();

    // Find contours
    std::vector<std::vector<cv::Point>> charsContrours = contourImage->_charsControus();

    // Loop on each object
    for (size_t i=0; i<charsContrours.size(); i++) {

        // Store points
        std::vector<cv::Point> points;
        for (size_t j = 0; j < charsContrours[i].size(); j++) {
            cv::Point p = charsContrours[i][j];
            points.push_back(p);
        }

        // Draw rectangle
        if(points.size() > 0){
            cv::Rect brect = cv::boundingRect(cv::Mat(points).reshape(2));
            cv::rectangle(*contourImage->getMat(), brect.tl() - cv::Point(2,2), brect.br()+cv::Point(2,2),
                          cv::Scalar(220, 100, 200), 1, CV_AA);
        }
    }
    return contourImage;
}

std::shared_ptr<Image> Image::_align(const std::vector<int> &indices,
                                     std::vector<std::vector<cv::Point>> &charsContours) {

    // Compute new sizes
    int padding = 2;
    int widthSum = padding *2;
    int heightSum = padding*2;
    for(int index=0; index < indices.size(); index++) {
        cv::Rect brect = cv::boundingRect(cv::Mat(charsContours[indices[index]]).reshape(2));
        widthSum += brect.width + padding;
        heightSum += brect.height;
    }

    // Determine new side value
    int side = std::max(widthSum, heightSum);

    // Construct and initialize a new mat
    std::shared_ptr<Image> image = _cloneImage();
    image->m_mat = std::make_shared<cv::Mat>(side, side, m_mat->type());
    image->getMat()->setTo(cv::Scalar(0));

    // Start populating the new matrix
    int leftOffset = padding;
    for(size_t i=0; i < indices.size(); i++) {

        // Get object
        cv::Rect brect = cv::boundingRect(cv::Mat(charsContours[indices[i]]).reshape(2));
        cv::Mat elementMat = ((*m_mat)(cv::Rect(brect.tl(), brect.br())));

        // Draw element on new matrix
        elementMat(cv::Rect(0,0,brect.width, brect.height)).copyTo(
                (*image->getMat())(cv::Rect(leftOffset, padding, brect.width, brect.height)));

        // Update left offset
        leftOffset += brect.width + padding;
    }
    return image;
}

void Image::_permutation(std::vector<std::shared_ptr<Image>> &outputImages, std::vector<int> &indices,
                         std::vector<std::vector<cv::Point>> &charsContours) {
    if(indices.size() == charsContours.size()) {
        std::shared_ptr<Image> newImage = _align(indices, charsContours);
        if(newImage) {
            outputImages.push_back(newImage);
        }
    } else {
        for(int i=0; i < charsContours.size(); i++) {
            if(std::find(indices.begin(), indices.end(), i) == indices.end()) {
                indices.push_back(i);
                _permutation(outputImages, indices, charsContours);
                indices.pop_back();
            }
        }
    }
}

std::vector<std::shared_ptr<Image>> Image::permutation() {
    std::vector<std::shared_ptr<Image>> perImages;
    std::vector<int> indices;
    std::vector<std::vector<cv::Point>> charsContours = _charsControus();
    _permutation(perImages, indices, charsContours);
    return perImages;
}

std::vector<std::shared_ptr<Image>> Image::split() {
    std::vector<std::shared_ptr<Image>> splitImages;
    std::vector<std::vector<cv::Point>> charsContours = _charsControus();
    for(int i=0; i < charsContours.size(); i++) {
        std::vector<int> indices = {i};
        std::shared_ptr<Image> elementImage = _align(indices, charsContours);
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
    std::shared_ptr<Image> rotImage = _cloneImage();
    cv::Point2f center(getSide()/2.0f, getSide()/2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Rect bbox = cv::RotatedRect(center,m_mat->size(), angle).boundingRect();
    rot.at<double>(0,2) += bbox.width/2.0 - center.x;
    rot.at<double>(1,2) += bbox.height/2.0 - center.y;
    cv::warpAffine(*m_mat, *rotImage->getMat(), rot, bbox.size());
    return rotImage;
}

std::vector<std::shared_ptr<Image>> Image::mnist() {
    // Prepare mnist vector
    std::vector<std::shared_ptr<Image>> mnistVector;
//    mnistVector.push_back(shared_from_this());

     // Invert colors
//    cv::bitwise_not (*m_mat, *m_mat);
//
     // Copy matrix to a larger one
//    cv::copyMakeBorder(*m_mat, *m_mat, 11,11,11,11, cv::BORDER_CONSTANT, cv::Scalar(0));
//
     // Rotate images
//    std::shared_ptr<Image> LRImage = rotate(45);
//    std::shared_ptr<Image> RRImage = rotate(-45);
//
     // Add new images
//    mnistVector.push_back(LRImage);
//    mnistVector.push_back(RRImage);
    return mnistVector;
}

std::shared_ptr<Image> Image::_cloneImage() {
    return std::make_shared<Image>(m_label, _cloneMat());
}

std::shared_ptr<Image> Image::size(int side) {
    std::shared_ptr<Image> image = _cloneImage();
    cv::resize(*image->getMat(), *image->getMat(), cv::Size(side, side));
    return image;
}

std::vector<charMatch> Image::extractChars() {
    cv::Mat inverse_img;
    cv::bitwise_not(*m_mat, inverse_img);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(inverse_img.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    std::vector<charMatch> result;

    for (int i(0); i < contours.size(); ++i) {
        cv::Rect bounding_box(cv::boundingRect(contours[i]));
//        int PADDING(2);
//        bounding_box.x -= PADDING;
//        bounding_box.y -= PADDING;
//        bounding_box.width += PADDING * 2;
//        bounding_box.height += PADDING * 2;

        charMatch match;
        cv::Point2i center(bounding_box.x + bounding_box.width / 2, bounding_box.y + bounding_box.height / 2);
        match.position = center;
        match.image = (*m_mat)(bounding_box);
        result.push_back(match);
    }

    std::sort(begin(result), end(result), [](charMatch const& a, charMatch const& b) -> bool {
        return a.position.x < b.position.x;
    });

    return result;
}

std::string Image::recognize(cv::Ptr<cv::ml::KNearest> kNN) {
    std::string result;
    if(kNN && kNN->isTrained()) {
        std::vector<charMatch> characters(extractChars());
        for (charMatch const& match : characters) {
            cv::Mat small_char;
            cv::resize(match.image, small_char, cv::Size(10, 10), 0, 0, cv::INTER_LINEAR);

            cv::Mat small_char_float;
            small_char.convertTo(small_char_float, CV_32FC1);

            cv::Mat small_char_linear(small_char_float.reshape(1, 1));

            cv::Mat response, distance;
            float p = kNN->findNearest(small_char_linear, kNN->getDefaultK(), cv::noArray(), response, distance);
//        std::cout << response << std::endl;
//        std::cout << distance << std::endl;

            result.push_back(char(p));
        }
    } else {
        std::cerr << "WARNING: Cannot recognize letter because KNN was not trained" << std::endl;
    }

    return result;
}

std::shared_ptr<cv::Mat> Image::_cloneMat() {
    std::shared_ptr<cv::Mat> mat = std::make_shared<cv::Mat>(getSide(), getSide(), m_mat->type());
    m_mat->copyTo(*mat);
    return mat;
}

std::vector<std::vector<cv::Point>> Image::_charsControus() {
    std::vector<std::vector<cv::Point>> charsContours;
    cv::findContours(*m_mat, charsContours, cv::RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    return charsContours;
}