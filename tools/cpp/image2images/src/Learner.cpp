#include <image2images/Learner.h>
#include <iostream>

void Learner::initKNN() {
    m_knn= cv::ml::KNearest::create();
    m_knn->setIsClassifier(true);
    m_knn->setAlgorithmType(cv::ml::KNearest::Types::BRUTE_FORCE);
    m_knn->setDefaultK(4);
}

std::pair<float, cv::Mat> Learner::_trainImage(char c, cv::Mat const& img) {

    // Make the image smaller
    cv::Mat smallMatrix;
    cv::resize(img, smallMatrix, cv::Size(m_trainingRows, m_trainingCols), 0, 0, cv::INTER_LINEAR);

    // Convert matrix to float
    cv::Mat smallMatrixFloat;
    smallMatrix.convertTo(smallMatrixFloat, CV_32FC1);

    // Make matrix flat
    cv::Mat smallMatrixFlat(smallMatrixFloat.reshape(1, 1));
    return std::pair<float, cv::Mat>(static_cast<float>(c), smallMatrixFlat);
}

bool Learner::trainKNN(std::string fileName) {
    std::ifstream knnInput(fileName);
    if(knnInput.is_open()) {

        // Extract features
        cv::Mat trainSamples, trainLabels;

        std::cout << ">> Loading training images ..." << std::endl;
        while(!knnInput.eof()) {
            cv::Mat char_img(28, 28, CV_8UC1);
            char label;
            knnInput >> label;
            for(int row=0; row < 28; row++) {
                for(int col=0; col < 28; col++) {
                    int val;
                    knnInput >> val;
                    char_img.at<uchar>(row, col) = (uchar)val;
                }
            }
            std::pair<float, cv::Mat> tinfo = _trainImage(label, char_img);
            trainLabels.push_back(tinfo.first);
            trainSamples.push_back(tinfo.second);
        }

        // Convert to train data
        cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(
                trainSamples, cv::ml::SampleTypes::ROW_SAMPLE, trainLabels);

        // Train kNN
        std::cout << ">> Started training kNN ..." << std::endl;
        m_knn->train(trainData);
        return m_knn->isTrained();
    } else {
        std::cerr << "Error opening kNN input file" << std::endl;
    }
    return false;
}