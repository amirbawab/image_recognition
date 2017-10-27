#include <iostream>
#include <fstream>
#include <unistd.h>
#include <getopt.h>
#include <image2images/File.h>
#include <image2images/Image.h>

// Global arguments
std::string g_inputFile;
int g_number = PIXLES;

/**
 * Print program usage to stdout
 */
void printUsage() {
    std::cout
            << "demo - Demo images" << std::endl
            << "    -i, --input     Input file" << std::endl
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
            {"help",   no_argument,       0, 'h'},
            {0, 0,                        0, 0}
    };

    int optionIndex = 0;
    int c;
    while ((c = getopt_long(argc, argv, "hi:n:", longOptions, &optionIndex)) != -1) {
        switch (c) {
            case 'i':
                g_inputFile = optarg;
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
    while (count++ < g_number) {
        std::shared_ptr<cv::Mat> mat = file.loadMat();
        std::shared_ptr<Image> image = std::make_shared<Image>(mat);
        images.push_back(image);
    }

    return 0;
}

