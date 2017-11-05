#include <image2images/Learner.h>
#include <iostream>

void Learner::initKNN() {
    m_knn= cv::ml::KNearest::create();
    m_knn->setIsClassifier(true);
    m_knn->setAlgorithmType(cv::ml::KNearest::Types::BRUTE_FORCE);
    m_knn->setDefaultK(5);
}

std::pair<float, cv::Mat> Learner::_prepareImage(std::shared_ptr<Image> image) {

    // Make the image smaller
    cv::Mat smallMatrix;
    cv::resize(*image->getMat(), smallMatrix, cv::Size(28, 28), 0, 0, cv::INTER_LINEAR);

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



        // Split data into train and test
//        trainData->setTrainTestSplitRatio(0.9);
//        std::cout << trainData->getNTestSamples() << "/" << trainData->getNTrainSamples() << std::endl;

        // Train kNN
        std::cout << ">> Started training kNN ..." << std::endl;
        m_knn->train(trainData);

        // Cross validation
//        cv::Mat testSamples = trainData->getTestSamples();
//        cv::Mat testResponses = trainData->getTestResponses();
//        int good = 0; int bad = 0;
//        for(int row=0; row < testSamples.rows; row++) {
//            float p = m_knn->findNearest(testSamples.row(row), m_knn->getDefaultK(), cv::noArray());
//            if(p == testResponses.at<float>(row, 0)) {
//                good++;
//            } else {
//                bad++;
//            }
//            std::cout << "Good: " << good << ", Bad: " << bad << ", Total samples: " << (good+bad) << testSamples.rows
//                      << ", Accuracy: " << (double) good / (good+bad) << std::endl;
//        }

        return m_knn->isTrained();
    } else {
        std::cerr << "Error opening training file: " << fileName << std::endl;
    }
    return false;
}

char Learner::findKNN(std::shared_ptr<Image> image) {
    if(m_knn && m_knn->isTrained()) {
        cv::Mat small_char;
        cv::resize(*image->getMat(), small_char, cv::Size(28, 28), 0, 0, cv::INTER_LINEAR);

        cv::Mat small_char_float;
        small_char.convertTo(small_char_float, CV_32FC1);

        cv::Mat small_char_linear(small_char_float.reshape(1, 1));

        cv::Mat response, distance;
        float p = m_knn->findNearest(small_char_linear, m_knn->getDefaultK(), cv::noArray(), response, distance);
//        std::cout << response << std::endl;
//        std::cout << distance << std::endl;

        if(p >= 0 && p <= 9) {
            return (char)(p + '0');
        } else if(p == 10) {
            return 'A';
        } else  {
            return 'M';
        }
    } else {
        std::cerr << "WARNING: Cannot recognize letter because KNN was not trained" << std::endl;
    }
    return '\u0000';
}

void Learner::validateKNN(std::vector<std::shared_ptr<Image>> images) {
    unsigned long NUM_ELEMENTS = 3;
    if(images.size() != NUM_ELEMENTS) {
        std::cout << "WARNING: Image has " << images.size()
                  << " instead of " << NUM_ELEMENTS << std::endl;
        m_bad++;
    } else {
        std::vector<char> elements;
        elements.push_back(findKNN(images[0]));
        elements.push_back(findKNN(images[1]));
        elements.push_back(findKNN(images[2]));

        std::vector<char> digits;
        std::vector<char> operations;
        for(char e : elements) {
            if(_isNum(e)) {
                digits.push_back(e);
            } else if(_isOperation(e)) {
                operations.push_back(e);
            } else {
                std::cerr << "Error validating kNN: Unknown digit or operation" << std::endl;
                m_bad++;
            }
        }

        int realLabel = images[0]->getLabel();
        if(digits.size() == 2 && operations.size() == 1) {
            if(operations[0] == 'A') {
                int label = digits[0] - '0' + digits[1] - '0';
                if(realLabel == label) {
                    m_good++;
                } else {
                    m_bad++;
                }
            } else if(operations[0] == 'M') {
                int label = (digits[0] - '0') * (digits[1] - '0');
                if(realLabel == label) {
                    m_good++;
                } else {
                    m_bad++;
                }
            } else {
                std::cout << "Error validation kNN: Unknown operation "<< operations[0] << std::endl;
                m_bad++;
            }
        } else {
            std::cout << "WARNING: Image has " << digits.size() << " digit(s) and "
                      << operations.size() << " operation(s)" << std::endl;
            m_bad++;
        }
    }
    std::cout << ">>>> Good: " << m_good << ", Bad: " << m_bad << ", Total: " << m_bad+m_good
              << ", Accuracy: " << (double) m_good / (m_good+m_bad) << std::endl;
}

bool Learner::_isNum(char a) {
    return a >= '0' && a <= '9';
}

bool Learner::_isOperation(char a) {
    return a == 'A' || a == 'M';
}