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

void Image::_reduceColors(int k) {
    int n = m_mat->rows * m_mat->cols;
    cv::Mat data = m_mat->reshape(1, n);
    data.convertTo(data, CV_32F);

    std::vector<int> labels;
    cv::Mat1f colors;
    cv::kmeans(data, k, labels
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
    binImage->m_mat = _cloneMat();

    // Enlarge
    int scale = 5;
    cv::resize(*binImage->getMat(), *binImage->getMat(),
               cv::Size(scale * binImage->getMat()->rows, scale * binImage->getMat()->cols));

    // Equalize
    cv::equalizeHist(*binImage->getMat(), *binImage->getMat());

    // Reduce colors
    _reduceColors(5);

    // Apply binary threshold
    cv::threshold(*binImage->getMat(), *binImage->getMat(), threshold, 255, CV_THRESH_BINARY);

    // Dilate
    int dilateVal = 4;
    cv::Mat kernel(cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilateVal, dilateVal)));
    cv::dilate(*binImage->getMat(), *binImage->getMat(), kernel, cv::Point(-1, -1), 1);

    // Recover original size
    cv::resize(*binImage->getMat(), *binImage->getMat(),
               cv::Size(binImage->getMat()->rows/scale, binImage->getMat()->cols/scale));
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
            startThreshold += 10;
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
    int padding = 3;
    int widthSum = 0;
    int height = 0;
    for(int index=0; index < indices.size(); index++) {
        cv::Rect brect = cv::boundingRect(cv::Mat(charsContours[indices[index]]).reshape(2));
        widthSum += std::max(brect.width, brect.height) + padding;
        height = std::max(height, brect.height);
    }

    // Add border padding
    widthSum += 2*padding;
    height += 2*padding;

    // Determine new side value
    int side = std::max(widthSum, height);

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

        // Create mask
        cv::Mat bigMask = cv::Mat::zeros(m_mat->rows, m_mat->cols, m_mat->type());
        drawContours(bigMask, charsContours, indices[i], cv::Scalar(255), CV_FILLED);
        cv::Mat smallMask = cv::Mat::zeros(image->getMat()->rows, image->getMat()->cols, image->getMat()->type());
        bigMask(cv::Rect(brect.x, brect.y, brect.width, brect.height)).copyTo(smallMask);

        // Compute top offset
        int topOffset = padding;
        if(side > brect.height) {
            topOffset = (side - brect.height) / 2;
        }

        // Compute left char offset
        int leftCharOffset = 0;
        if(brect.height > brect.width) {
            leftCharOffset = (brect.height - brect.width) / 2;
        }

        // Draw element on new matrix
        elementMat.copyTo(
                (*image->getMat())(cv::Rect(leftOffset + leftCharOffset, topOffset, brect.width, brect.height)), smallMask);

        // Update left offset
        leftOffset += std::max(brect.width, brect.height) + padding;
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
    mnistVector.push_back(shared_from_this());

     // Rotate images
    std::shared_ptr<Image> LRImage = rotate(45);
    std::shared_ptr<Image> RRImage = rotate(-45);

     // Add new images
    mnistVector.push_back(LRImage);
    mnistVector.push_back(RRImage);
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

std::shared_ptr<Image> Image::erode(int size) {
    std::shared_ptr<Image> image = _cloneImage();
    cv::erode(*image->getMat(), *image->getMat(), cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(size, size)));
    return image;
}

std::vector<charMatch> Image::extractChars() {
    std::vector<std::vector<cv::Point>> contours = _charsControus();
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
            cv::resize(match.image, small_char, cv::Size(28, 28), 0, 0, cv::INTER_LINEAR);

            cv::Mat small_char_float;
            small_char.convertTo(small_char_float, CV_32FC1);

            cv::Mat small_char_linear(small_char_float.reshape(1, 1));

            cv::Mat response, distance;
            float p = kNN->findNearest(small_char_linear, kNN->getDefaultK(), cv::noArray(), response, distance);
//        std::cout << response << std::endl;
//        std::cout << distance << std::endl;

            result.push_back((char)(p + '0'));
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