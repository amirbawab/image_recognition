#include <image2images/Learner.h>
#include <iostream>

void Learner::initKNN() {
    m_knn= cv::ml::KNearest::create();
    m_knn->setIsClassifier(true);
    m_knn->setAlgorithmType(cv::ml::KNearest::Types::BRUTE_FORCE);
    m_knn->setDefaultK(4);
}

std::pair<float, cv::Mat> Learner::_prepareImage(std::shared_ptr<Image> image) {

    // Make the image smaller
    cv::Mat smallMatrix;
    cv::resize(*image->getMat(), smallMatrix, cv::Size(10, 10), 0, 0, cv::INTER_LINEAR);

    // Convert matrix to float
    cv::Mat smallMatrixFloat;
    smallMatrix.convertTo(smallMatrixFloat, CV_32FC1);

    // Make matrix flat
    cv::Mat smallMatrixFlat(smallMatrixFloat.reshape(1, 1));
    return std::pair<float, cv::Mat>(static_cast<float>(image->getLabel()), smallMatrixFlat);
}

bool Learner::trainKNN(std::string fileName) {

    std::cout << ">> Loading training images" << std::endl;
    File knnFile;
    if(knnFile.read(fileName, 0)) {

        // Extract features
        cv::Mat trainSamples, trainLabels;
        std::cout << ">> Processing training images ..." << std::endl;
        for(int i=0; i < knnFile.getSize(); i++) {
            std::pair<float, cv::Mat> metaMat = _prepareImage(knnFile.getImage(i));
            trainLabels.push_back(metaMat.first);
            trainSamples.push_back(metaMat.second);
        }

        // Convert to train data
        cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(
                trainSamples, cv::ml::SampleTypes::ROW_SAMPLE, trainLabels);

        // Train kNN
        std::cout << ">> Started training kNN ..." << std::endl;
        m_knn->train(trainData);
        return m_knn->isTrained();
    } else {
        std::cerr << "Error opening training file: " << fileName << std::endl;
    }
    return false;
}