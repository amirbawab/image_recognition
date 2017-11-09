#include <iostream>
#include <fstream>
#include <unistd.h>
#include <getopt.h>
#include <image2images/File.h>
#include <image2images/Image.h>
#include <fstream>
#include <image2images/Learner.h>

// Codes
const int CODE_ERROR = 1;

// Global arguments
std::vector<std::string> g_algos;
std::string g_inputFile;
std::string g_outputDir;
std::string g_knnFile;
std::string g_cnnFile;
unsigned int g_number = 1;
unsigned int g_offset = 0;
bool g_display = false;
bool g_matrix = false;
bool g_cmatrix = false;

// Learner variable
Learner g_learner;

// Algorithms
const std::string ALGO_BINARY =         "binary";
const std::string ALGO_PERMUTATION =    "permutation";
const std::string ALGO_CLEAN =          "clean";
const std::string ALGO_CONTOUR =        "contour";
const std::string ALGO_ALIGN =          "align";
const std::string ALGO_SPLIT =          "split";
const std::string ALGO_ROTATE =         "rotate";
const std::string ALGO_MNIST =          "mnist";
const std::string ALGO_SIZE =           "size";
const std::string ALGO_BLUR =           "blur";
const std::string ALGO_FINDKNN =        "findKNN";
const std::string ALGO_VALIDATEKNN =    "validateKNN";
const std::string ALGO_RUNKNN =         "runKNN";
const std::string ALGO_VALIDATECNN =    "validateCNN";

/**
 * Print program usage to stdout
 */
void printUsage() {
    std::cout
            << "image2images - Process images" << std::endl
            << "    -i, --input      Input file" << std::endl
            << "    -a, --algorithm  Algorithms" << std::endl
            << "                     - " << ALGO_BINARY << "{1..255}: Convert image to binary with the "
                                                        << "provided threshold" << std::endl
            << "                     - " << ALGO_CLEAN << ": Clean noise in image" << std::endl
            << "                     - " << ALGO_PERMUTATION << ": Generate new permutation images - DEPRECATED" << std::endl
            << "                     - " << ALGO_CONTOUR << ": Draw contour around objects" << std::endl
            << "                     - " << ALGO_ALIGN << "{1,3}: Align detected elements in image" << std::endl
            << "                     - " << ALGO_SPLIT << ": Generate an image per detected element" << std::endl
            << "                     - " << ALGO_ROTATE << "{1..360}: Rotate images by the provided angle" << std::endl
            << "                     - " << ALGO_MNIST << ": Algorithm optimized for MNIST dataset" << std::endl
            << "                     - " << ALGO_SIZE << "{1..N}: Set image size" << std::endl
            << "                     - " << ALGO_FINDKNN << ": Recognize image using kNN" << std::endl
            << "                     - " << ALGO_VALIDATECNN << ": Validate CNN from CNN input file" << std::endl
            << "                     - " << ALGO_VALIDATEKNN << ": Validate the input. "
                                                            << "Must use only if labels are known" << std::endl
            << "                     - " << ALGO_RUNKNN << ": Write kNN results to file. "
                                                            << "Must use for unknown labels" << std::endl
            << "    -o, --output     Output directory" << std::endl
            << "    -m, --matrix     Output as matrix instead of image" << std::endl
            << "    -M, --cmatrix    Same as --matrix but all in one file" << std::endl
            << "    -d, --display    Show images in windows" << std::endl
            << "    -s, --offset     Offset/Starting image" << std::endl
            << "    -c, --cnn        CNN file (Format: 1 2 A)" << std::endl
            << "    -n, --number     Number of images to show" << std::endl
            << "    -k, --knn        Path for the kNN file" << std::endl
            << "    -h, --help       Display this help message" << std::endl;
}

/**
 * Initialize parameters
 * @param argc
 * @param argv
 */
void initParams(int argc, char *argv[]) {
    struct option longOptions[] = {
            {"input", required_argument,  0, 'i'},
            {"output", required_argument, 0, 'o'},
            {"algorithm", required_argument, 0, 'a'},
            {"number", required_argument, 0, 'n'},
            {"offset", required_argument, 0, 's'},
            {"display",   no_argument,       0, 'd'},
            {"matrix",   no_argument,       0, 'm'},
            {"cmatrix",   no_argument,       0, 'M'},
            {"cnn",   required_argument,       0, 'c'},
            {"knn",   required_argument,       0, 'k'},
            {"help",   no_argument,       0, 'h'},
            {0, 0,                        0, 0}
    };

    int optionIndex = 0;
    int c;
    while ((c = getopt_long(argc, argv, "ho:i:n:s:a:dmk:Mc:", longOptions, &optionIndex)) != -1) {
        switch (c) {
            case 'i':
                g_inputFile = optarg;
                break;
            case 's':
                g_offset = (unsigned int) atoi(optarg);
                break;
            case 'o':
                g_outputDir= optarg;
                break;
            case 'a':
                g_algos.push_back(optarg);
                break;
            case 'm':
                g_matrix = true;
                break;
            case 'M':
                g_cmatrix = true;
                break;
            case 'd':
                g_display = true;
                break;
            case 'n':
                g_number = (unsigned int) atoi(optarg);
                break;
            case 'k':
                g_knnFile = optarg;
                break;
            case 'c':
                g_cnnFile = optarg;
                break;
            case 'h':
            default:
                break;
        }
    }
}

int main( int argc, char** argv ) {
    // Initialize parameters
    initParams(argc, argv);

    // Check for missing params
    if(g_inputFile.empty()) {
        printUsage();
        return 0;
    }

    // Train kNN
    if(!g_knnFile.empty()) {
        g_learner.initKNN();
        if(g_learner.trainKNN(g_knnFile)) {
            std::cout << ">> Training kNN completed!" << std::endl;
        } else {
            std::cerr << "Error training kNN" << std::endl;
        }
    }

    File file;
    if(!file.read(g_inputFile, g_number + g_offset)) {
        std::cerr << "Error opening input file: " << g_inputFile << std::endl;
        return CODE_ERROR;
    }

    // Read cnn file
    std::ifstream cnnFile;
    if(!g_cnnFile.empty()) {
        cnnFile.open(g_cnnFile);
    }

    // Clear tmp file for cmatrix
    std::string cmatrixFile = g_outputDir + "/all_ocv.ocv";
    std::string cmatrixTmpFile = cmatrixFile + ".tmp";
    if(g_cmatrix) {
        std::ofstream matrixTmpFile(cmatrixTmpFile, std::ofstream::out | std::ofstream::trunc);
        if(matrixTmpFile.is_open()) {
            matrixTmpFile.close();
        }
    }

    // Process target images
    std::cout << ">> Starting image processing ..." << std::endl;
    bool windowOpen = false;
    long totalMats = 0;
    for(int imageIndex = g_offset; imageIndex < file.getSize(); imageIndex++){

        // Get image and check if it does exist
        std::shared_ptr<Image> image = file.getImage(imageIndex);
        if(!image) {
            std::cout << "Matrix at index: " << imageIndex << " was not found. Halting ..." << std::endl;
            exit(1);
        }

        // Store progress
        int progress = imageIndex - g_offset;

        // Prepare vector for the images to generate
        std::vector<std::shared_ptr<Image>> outputImages;
        outputImages.push_back(image);

        // Apply algorithms
        for(std::string algo : g_algos) {
            if(algo == ALGO_BINARY) {
                std::vector<std::shared_ptr<Image>> manipOutputImages;
                for(auto outputImage : outputImages) {
                    std::shared_ptr<Image> binImage = outputImage->binarize();
                    manipOutputImages.push_back(binImage);
                }
                outputImages = manipOutputImages;
            } else if(algo == ALGO_PERMUTATION) {
                std::vector<std::shared_ptr<Image>> manipOutputImages;
                for(auto outputImage : outputImages) {
                    std::vector<std::shared_ptr<Image>> per = outputImage->permutation();
                    manipOutputImages.insert(manipOutputImages.end(), per.begin(), per.end());
                }
                outputImages = manipOutputImages;
            } else if(algo == ALGO_CLEAN) {
                std::vector<std::shared_ptr<Image>> manipOutputImages;
                for (auto outputImage : outputImages) {
                    std::shared_ptr<Image> cleanImage = outputImage->cleanNoise();
                    manipOutputImages.push_back(cleanImage);
                }
                outputImages = manipOutputImages;
            } else if(algo.rfind(ALGO_ALIGN, 0) == 0 && algo.size() > ALGO_ALIGN.size()) {
                int val = atoi(algo.substr(ALGO_ALIGN.size(), algo.size() - ALGO_ALIGN.size()).c_str());
                std::vector<std::shared_ptr<Image>> manipOutputImages;
                for (auto outputImage : outputImages) {
                    std::shared_ptr<Image> align = outputImage->align(val);
                    manipOutputImages.push_back(align);
                }
                outputImages = manipOutputImages;
            } else if(algo.rfind(ALGO_ROTATE, 0) == 0 && algo.size() > ALGO_ROTATE.size()) {
                int val = atoi(algo.substr(ALGO_ROTATE.size(), algo.size() - ALGO_ROTATE.size()).c_str());
                std::vector<std::shared_ptr<Image>> manipOutputImages;
                for (auto outputImage : outputImages) {
                    std::shared_ptr<Image> rotImage = outputImage->rotate(val);
                    manipOutputImages.push_back(rotImage);
                }
                outputImages = manipOutputImages;
            } else if(algo == ALGO_SPLIT) {
                std::vector<std::shared_ptr<Image>> manipOutputImages;
                for(auto outputImage : outputImages) {
                    std::vector<std::shared_ptr<Image>> split = outputImage->split();
                    manipOutputImages.insert(manipOutputImages.end(), split.begin(), split.end());
                }
                outputImages = manipOutputImages;
            } else if(algo.rfind(ALGO_CONTOUR, 0) == 0 && algo.size() > ALGO_CONTOUR.size()) {
                int numContours = atoi(algo.substr(ALGO_CONTOUR.size(), algo.size() - ALGO_CONTOUR.size()).c_str());
                std::vector<std::shared_ptr<Image>> manipOutputImages;
                for (auto outputImage : outputImages) {
                    std::shared_ptr<Image> contourImage = outputImage->drawContour(numContours);
                    manipOutputImages.push_back(contourImage);
                }
                outputImages = manipOutputImages;
            } else if(algo == ALGO_MNIST) {
                std::vector<std::shared_ptr<Image>> manipOutputImages;
                for (auto outputImage : outputImages) {
                    std::vector<std::shared_ptr<Image>> mnist = outputImage->mnist();
                    manipOutputImages.insert(manipOutputImages.end(), mnist.begin(), mnist.end());
                }
                outputImages = manipOutputImages;
            } else if(algo.rfind(ALGO_SIZE, 0) == 0 && algo.size() > ALGO_SIZE.size()) {
                int side = atoi(algo.substr(ALGO_SIZE.size(), algo.size() - ALGO_SIZE.size()).c_str());
                std::vector<std::shared_ptr<Image>> manipOutputImages;
                for (auto outputImage : outputImages) {
                    std::shared_ptr<Image> scaledImage = outputImage->size(side);
                    manipOutputImages.push_back(scaledImage);
                }
                outputImages = manipOutputImages;
            } else if(algo == ALGO_BLUR) {
                std::vector<std::shared_ptr<Image>> manipOutputImages;
                for (auto outputImage : outputImages) {
                    std::shared_ptr<Image> erodeImage = outputImage->blur();
                    manipOutputImages.push_back(erodeImage);
                }
                outputImages = manipOutputImages;
            } else if(algo == ALGO_FINDKNN) {
                for (auto outputImage : outputImages) {
                    char element = g_learner.findKNN(outputImage);
                    std::cout << "Image ID: " << outputImage->getId() << " >> " << element << std::endl;
                }
            } else if(algo == ALGO_VALIDATECNN) {
                if(cnnFile.is_open()) {
                    for (auto outputImage : outputImages) {
                        int a, b, c;
                        cnnFile >> a >> b >> c;
                        std::vector<char> labels;
                        if(a < 10) {
                            labels.push_back((char)(a+'0'));
                        } else if(a == 10) {
                            labels.push_back('A');
                        } else {
                            labels.push_back('M');
                        }
                        if(b < 10) {
                            labels.push_back((char)(b+'0'));
                        } else if(b == 10) {
                            labels.push_back('A');
                        } else {
                            labels.push_back('M');
                        }
                        if(c < 10) {
                            labels.push_back((char)(c+'0'));
                        } else if(c == 10) {
                            labels.push_back('A');
                        } else {
                            labels.push_back('M');
                        }
                        g_learner.validateCNN(labels, outputImage->getLabel(), progress);
                    }
                } else {
                    std::cerr << "Error: CNN file not open" << std::endl;
                }
            } else if(algo == ALGO_VALIDATEKNN) {
                g_learner.validateKNN(outputImages, progress);
            } else if(algo == ALGO_RUNKNN) {
                g_learner.runKNN("/tmp/result_knn.csv", progress+1, outputImages);
            } else {
                std::cerr << "Algorithm " << algo << " not found!" << std::endl;
            }
        }

        // Generate names
        for(auto outputImage : outputImages) {
            std::stringstream name;
            name << progress+1 << "_" << outputImage->getId();
            outputImage->setName(name.str());
        }

        // Generate output
        if(!g_outputDir.empty()) {
            for(auto outputImage : outputImages) {
                std::stringstream fileName;
                fileName << g_outputDir << "/" << outputImage->getLabel() << "/" << outputImage->getName();
                if(g_matrix) {
                    fileName << ".ocv";
                    std::cout << ">> Generating matrix: " << fileName.str() << std::endl;
                    std::ofstream matrixFile(fileName.str());
                    if(!matrixFile.is_open()) {
                        std::cerr << "Error generating matrix: " << fileName.str() << std::endl;
                    } else {
                        matrixFile << outputImage->getLabel() << " " << outputImage->getSide();
                        for(int row=0; row < outputImage->getMat()->rows; row++) {
                            for(int col=0; col < outputImage->getMat()->cols; col++) {
                                matrixFile << " " << (int) outputImage->getMat()->at<uchar>(row, col);
                            }
                        }
                        matrixFile.close();
                    }
                } else if(g_cmatrix) {
                    std::cout << ">> Adding matrix " << outputImage->getName() << " to output file "
                              << cmatrixTmpFile << std::endl;
                    std::ofstream matrixFile(cmatrixTmpFile, std::ios::app);
                    if(!matrixFile.is_open()) {
                        std::cerr << "Error generating matrix: " << cmatrixTmpFile << std::endl;
                    } else {
                        matrixFile << outputImage->getLabel() << " " << outputImage->getSide();
                        for(int row=0; row < outputImage->getMat()->rows; row++) {
                            for(int col=0; col < outputImage->getMat()->cols; col++) {
                                matrixFile << " " << (int) outputImage->getMat()->at<uchar>(row, col);
                            }
                        }
                        matrixFile << std::endl;
                        matrixFile.close();
                        totalMats++;
                    }

                } else {
                    fileName << ".png";
                    std::cout << ">> Generating image: " << fileName.str() << std::endl;
                    if(!cv::imwrite(fileName.str(), *outputImage->getMat())) {
                        std::cerr << "Error generating image: " << fileName.str() << std::endl;
                    }
                }
            }
        }

        // Display output images
        if(g_display) {
            for(auto outputImage : outputImages) {
                std::cout << ">> Displaying image id: " << outputImage->getName() << std::endl;
                outputImage->display();
                windowOpen = true;
            }
        }

        // Log
        if( progress % 100 == 0) {
            std::cout << ">> Processed " << progress + 1 << " images out of " << g_number << std::endl;
        }
    }

    // Rewrite cmatrix file
    if(g_cmatrix) {
        std::cout << ">> Generating output file: " << cmatrixFile << std::endl;
        std::ofstream matrixFile(cmatrixFile);
        std::ifstream matrixTmpFile(cmatrixTmpFile);
        if (!matrixFile.is_open()) {
            std::cerr << "Error generating matrix: " << cmatrixFile << std::endl;
        } else {
            matrixFile << totalMats << std::endl;
            matrixFile << matrixTmpFile.rdbuf();
        }
        matrixFile.close();
        matrixTmpFile.close();
        std::remove(cmatrixTmpFile.c_str());
    }


    // Pause if new images found
    if(windowOpen) {
        Image::wait();
    }

    return 0;
}

