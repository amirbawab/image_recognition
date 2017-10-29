#include <iostream>
#include <fstream>
#include <unistd.h>
#include <getopt.h>
#include <image2images/File.h>
#include <image2images/Image.h>

// Global arguments
std::string g_inputFile;
int g_number = 1;
int g_offset = 0;

/**
 * Print program usage to stdout
 */
void printUsage() {
    std::cout
            << "image2images - Process images" << std::endl
            << "    -i, --input     Input file" << std::endl
            << "    -s, --offset    Offset/Starting image" << std::endl
            << "    -n, --number    Number of images to show" << std::endl
            << "    -h, --help      Display this help message" << std::endl;
}

/**
 * Initialize parameters
 * @param argc
 * @param argv
 */
void initParams(int argc, char *argv[]) {
    struct option longOptions[] = {
            {"input", required_argument, 0, 'i'},
            {"number", required_argument, 0, 'n'},
            {"offset", required_argument, 0, 's'},
            {"help",   no_argument,       0, 'h'},
            {0, 0,                        0, 0}
    };

    int optionIndex = 0;
    int c;
    while ((c = getopt_long(argc, argv, "hi:n:s:", longOptions, &optionIndex)) != -1) {
        switch (c) {
            case 'i':
                g_inputFile = optarg;
                break;
            case 's':
                g_offset = atoi(optarg);
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
        // Convert image to binary
        image->binarize();

        // Extract objects
        image->extract();

        // Clean noise objects
        image->cleanNoise();

        // Draw contour around objects
        image->contour();

        // Display matrix
        image->display();
    }
    Image::wait();

    return 0;
}

