#pragma once

#include <image2images/File.h>
#include <image2images/Image.h>

class Learner {
private:
    cv::Ptr<cv::ml::KNearest> m_knn;

    /**
     * Convert image to a vector for training
     * @param image
     * @return tuple of the processed matrix and its lable
     */
    std::pair<float, cv::Mat> _prepareImage(std::shared_ptr<Image> image);
public:

    /**
     * Initialize kNN
     */
    void initKNN();

    /**
     * Train kNN
     * @param fileName
     */
    bool trainKNN(std::string fileName);

    /**
     * Get kNN instance
     * @return kNN
     */
    cv::Ptr<cv::ml::KNearest> getKnn() const {return m_knn;}
};