#include <iostream>
#include <fstream>
#include <unistd.h>
#include <getopt.h>
#include <image2images/File.h>
#include <image2images/Image.h>

// Global arguments
std::vector<std::string> g_algos;
std::string g_inputFile;
std::string g_outputFile;
int g_number = 1;
int g_offset = 0;
bool g_display = false;

// Algorithms
const std::string ALGO_BINARY =         "binary";
const std::string ALGO_PERMUTATION =    "permutation";
const std::string ALGO_CLEAN =          "clean";
const std::string ALGO_CONTOUR =        "contour";
const std::string ALGO_DETECT =         "detect";

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
            << "    -o, --output     Output file" << std::endl
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
            {"display",   no_argument,       0, 'd'},
            {"help",   no_argument,       0, 'h'},
            {0, 0,                        0, 0}
    };

    int optionIndex = 0;
    int c;
    while ((c = getopt_long(argc, argv, "ho:i:n:s:a:d", longOptions, &optionIndex)) != -1) {
        switch (c) {
            case 'i':
                g_inputFile = optarg;
                break;
            case 's':
                g_offset = atoi(optarg);
                break;
            case 'o':
                g_outputFile= optarg;
                break;
            case 'a':
                g_algos.push_back(optarg);
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

    // Load file
    File file(g_inputFile);
    int count = 0;

    // Create and store all images
    std::vector<std::shared_ptr<Image>> images;
    while (count < g_offset + g_number) {
        if(count < g_offset) {
            file.skipMat();
        } else {
            std::shared_ptr<cv::Mat> mat = file.loadMat();
            std::shared_ptr<Image> image = std::make_shared<Image>(mat);
            images.push_back(image);
        }
        count++;
    }

    // Adjust all images
    for(auto image : images){

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
                for(auto outputImage : outputImages) {
                    outputImages = image->permutation();
                }
            } else if(algo == ALGO_DETECT){
                for(auto outputImage : outputImages) {
                    outputImage->detectElements();
                }
            } else if(algo == ALGO_CLEAN) {
                for(auto outputImage : outputImages) {
                    outputImage->cleanNoise();
                }
            } else if(algo == ALGO_CONTOUR) {
                for(auto outputImage : outputImages) {
                    outputImage->contour();
                }
            } else {
                std::cerr << "Algorithm " << algo << " was not found!" << std::endl;
            }
        }

        // Display output images
        if(g_display) {
            for(auto outputImage : outputImages) {
                outputImage->display();
            }
            // Pause if new images found
            if(!outputImages.empty()) {
                Image::wait();
            }
        }
    }

    return 0;
}

