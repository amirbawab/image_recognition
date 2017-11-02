#pragma once

#include <image2images/File.h>
#include <image2images/Image.h>

class Learner {
private:
    cv::Ptr<cv::ml::KNearest> m_knn;
    int m_trainingRows = 15;
    int m_trainingCols = 15;

    /**
     * Convert image to a vector for training
     * @param c
     * @param img
     * @return tuple of the processed matrix and its lable
     */
    std::pair<float, cv::Mat> _trainImage(char c, cv::Mat const& img);
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