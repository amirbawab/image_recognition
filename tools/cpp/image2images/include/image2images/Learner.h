#pragma once

#include <functional>
#include <image2images/File.h>
#include <image2images/Image.h>
#include <neuralnetwork/NeuralNetwork.h>

class Learner {
private:
    cv::Ptr<cv::ml::KNearest> m_knn;
    std::shared_ptr<Network> m_network;
    int m_good = 0;
    int m_bad = 0;

    /**
     * Convert image to a vector for training
     * @param image
     * @return tuple of the processed matrix and its lable
     */
    std::pair<float, cv::Mat> _prepareImage(std::shared_ptr<Image> image);

    /**
     * Check if character is a number
     * @param a
     * @return true if it is a number
     */
    bool _isNum(char a);

    /**
     * Check if character is an addition or multiplication
     * @param a
     * @return true if it is an addition or multiplication
     */
    bool _isOperation(char a);

    /**
     * Validate kNN
     * @return result
     */
    int _getLabel(std::vector<std::shared_ptr<Image>> images, int id, std::function<std::vector<char>()> algoFunc);
public:

    /**
     * Initialize Neural Network
     */
    void initNN();

    /**
     * Train neural network
     * @param fileName
     * @return
     */
    bool trainNN(std::string fileName);

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

    /**
     * Find kNN in image
     * @param kNN
     * @return recognize images
     */
    char findKNN(std::shared_ptr<Image> image);

    /**
     * Validate images
     */
    void validateKNN(std::vector<std::shared_ptr<Image>> images, int id);

    /**
     * Validate chars
     */
    void validateCNN(std::vector<char> labels, int realLabel, int id);

    /**
     * Validate chars
     */
    void validateNN(std::shared_ptr<Image> image, int id);

    /**
     * Write kNN results to file
     * @param fileName
     * @param id
     * @param images
     */
    void runKNN(std::string fileName, int id, std::vector<std::shared_ptr<Image>> images);
};