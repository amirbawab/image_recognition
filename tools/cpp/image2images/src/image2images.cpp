#include <iostream>
#include <fstream>
#include <unistd.h>
#include <getopt.h>
#include <image2images/File.h>
#include <image2images/Image.h>
#include <fstream>

// Codes
const int CODE_ERROR = 1;

// Global arguments
std::vector<std::string> g_algos;
std::string g_inputFile;
std::string g_outputDir;
std::string g_labelFile;
int g_number = 1;
int g_offset = 0;
bool g_display = false;
bool g_matrix = false;

// Algorithms
const std::string ALGO_BINARY =         "binary";
const std::string ALGO_PERMUTATION =    "permutation";
const std::string ALGO_CLEAN =          "clean";
const std::string ALGO_CONTOUR =        "contour";
const std::string ALGO_DETECT =         "detect";
const std::string ALGO_ALIGN =          "align";
const std::string ALGO_SPLIT =          "split";

/**
 * Print program usage to stdout
 */
void printUsage() {
    std::cout
            << "image2images - Process images" << std::endl
            << "    -i, --input      Input file" << std::endl
            << "    -a, --algorithm  Algorithms" << std::endl
            << "                     - " << ALGO_BINARY << ": Convert image to binary" << std::endl
            << "                     - " << ALGO_CLEAN << ": Clean noise in image" << std::endl
            << "                     - " << ALGO_PERMUTATION << ": Generate new permutation images" << std::endl
            << "                     - " << ALGO_CONTOUR << ": Draw contour around objects" << std::endl
            << "                     - " << ALGO_DETECT << ": Detect elements in image" << std::endl
            << "                     - " << ALGO_ALIGN << ": Align detected elements in image" << std::endl
            << "                     - " << ALGO_SPLIT << ": Generate an image per detected element" << std::endl
            << "    -o, --output     Output directory" << std::endl
            << "    -m, --matrix     Output as matrix instead of image" << std::endl
            << "    -l, --label      Label file" << std::endl
            << "    -d, --display    Show images in windows" << std::endl
            << "    -s, --offset     Offset/Starting image" << std::endl
            << "    -n, --number     Number of images to show" << std::endl
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
            {"label", required_argument, 0, 'l'},
            {"display",   no_argument,       0, 'd'},
            {"matrix",   no_argument,       0, 'm'},
            {"help",   no_argument,       0, 'h'},
            {0, 0,                        0, 0}
    };

    int optionIndex = 0;
    int c;
    while ((c = getopt_long(argc, argv, "ho:i:n:s:a:dml:", longOptions, &optionIndex)) != -1) {
        switch (c) {
            case 'i':
                g_inputFile = optarg;
                break;
            case 's':
                g_offset = atoi(optarg);
                break;
            case 'l':
                g_labelFile = optarg;
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
            case 'd':
                g_display = true;
                break;
            case 'n':
                g_number = atoi(optarg);
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

    // Check if input file was found
    File file;
    if(!file.setInputFile(g_inputFile)) {
        std::cerr << "Error opening input file: " << g_inputFile << std::endl;
        return CODE_ERROR;
    }

    // Check if label file was found
    if(!g_labelFile.empty() && !file.setLabelFile(g_labelFile)) {
        std::cerr << "Error opening csv file: " << g_labelFile << std::endl;
        return CODE_ERROR;
    }

    // Create and store all images
    int count = 0;
    std::vector<std::shared_ptr<Image>> images;
    std::cout << "Loading images ..." << std::endl;
    while (count < g_offset + g_number) {
        if(count < g_offset) {
            file.skipMat();
        } else {
            images.push_back(file.loadImage());
        }
        count++;

        // Log
        int progress = count - g_offset;
        if(progress > 0 && progress % 100 == 0) {
            std::cout << ">> Loaded " << progress << " images out of " << g_number << std::endl;
        }
    }

    // Adjust all images
    std::cout << "Starting image processing ..." << std::endl;
    int progress = 0;
    bool windowOpen = false;
    for(std::shared_ptr<Image> image : images){

        // Prepare vector for the images to generate
        std::vector<std::shared_ptr<Image>> outputImages;
        outputImages.push_back(image);

        // Apply algorithms
        for(std::string algo : g_algos) {
            if(algo == ALGO_BINARY) {
                for(auto outputImage : outputImages) {
                    outputImage->binarize();
                }
            } else if(algo == ALGO_PERMUTATION) {
                std::vector<std::shared_ptr<Image>> manipOutputImages;
                for(auto outputImage : outputImages) {
                    std::vector<std::shared_ptr<Image>> per = image->permutation();
                    manipOutputImages.insert(manipOutputImages.end(), per.begin(), per.end());
                }
                outputImages = manipOutputImages;
            } else if(algo == ALGO_DETECT){
                for(auto outputImage : outputImages) {
                    outputImage->detectElements();
                }
            } else if(algo == ALGO_CLEAN) {
                for (auto outputImage : outputImages) {
                    outputImage->cleanNoise();
                }
            } else if(algo == ALGO_ALIGN) {
                std::vector<std::shared_ptr<Image>> manipOutputImages;
                for (auto outputImage : outputImages) {
                    std::shared_ptr<Image> align = outputImage->align();
                    // TODO Handle case of image cannot be created
                    if (align) {
                        manipOutputImages.push_back(align);
                    }
                }
                outputImages = manipOutputImages;
            } else if(algo == ALGO_SPLIT) {
                std::vector<std::shared_ptr<Image>> manipOutputImages;
                for(auto outputImage : outputImages) {
                    std::vector<std::shared_ptr<Image>> split = outputImage->split();
                    manipOutputImages.insert(manipOutputImages.end(), split.begin(), split.end());
                }
                outputImages = manipOutputImages;
            } else if(algo == ALGO_CONTOUR) {
                for(auto outputImage : outputImages) {
                    outputImage->contour();
                }
            }
        }

        // Generate output
        if(!g_outputDir.empty()) {
            for(auto outputImage : outputImages) {
                std::stringstream fileName;
                fileName << g_outputDir << "/" << outputImage->getValue() << "/" << progress << "_"
                         << outputImage->getId();
                if(!g_matrix) {
                    fileName << ".jpg";
                    std::cout << ">> Generating image: " << fileName.str() << std::endl;
                    if(!cv::imwrite(fileName.str(), *outputImage->getMat())) {
                        std::cerr << "Error generating image: " << fileName.str() << std::endl;
                    }
                } else {
                    fileName << ".csv";
                    std::cout << ">> Generating matrix: " << fileName.str() << std::endl;
                    std::ofstream matrixFile(fileName.str());
                    if(!matrixFile.is_open()) {
                        std::cerr << "Error generating matrix: " << fileName.str() << std::endl;
                    } else {
                        for(int row=0; row < outputImage->getMat()->rows; row++) {
                            for(int col=0; col < outputImage->getMat()->cols; col++) {
                                if(row != 0 || col != 0) {
                                    matrixFile << ",";
                                }
                                matrixFile << (int)outputImage->getMat()->at<uchar>(row, col);
                            }
                        }
                        matrixFile.close();
                    }
                }
            }
        }

        // Display output images
        if(g_display) {
            for(auto outputImage : outputImages) {
                std::cout << ">> Displaying image id: " << outputImage->getId() << std::endl;
                outputImage->display();
                windowOpen = true;
            }
        }

        // Log
        if(++progress % 100 == 0) {
            std::cout << ">> Processed " << progress << " images out of " << images.size() << std::endl;
        }
    }

    // Pause if new images found
    if(windowOpen) {
        Image::wait();
    }

    return 0;
}

