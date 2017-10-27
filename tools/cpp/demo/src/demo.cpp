#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <unistd.h>
#include <getopt.h>

#define ROWS 64
#define COLS 64
#define PIXLES ROWS * COLS

// Global arguments
std::string g_inputFile;
int g_number = 1;

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
 * Read a matrix from a file
 * @param input
 * @return matrix
 */
cv::Mat file2Mat(std::ifstream &input) {
    cv::Mat out(ROWS, COLS, CV_8UC1);
    int row = 0;
    int col = 0;
    for(int i=0; i < PIXLES; i++) {

        // Load input from file
        double val;
        input >> val;

        // Update pixel
        out.at<uchar>(row, col) = (uchar)(255 - val);
        if(++col == COLS) {
            row++;
            col = 0;
        }
    }
    return out;
}

/**
 * Show images
 * @param fileName
 * @param images
 */
void showImages(std::string fileName, int images) {
    std::ifstream input(fileName);

    // Set image configuration
    if(input.is_open()) {
        while(images-- > 0) {

            // Load matrix from file
            cv::Mat out = file2Mat(input);

            // Set a unique window name
            std::stringstream winName;
            winName << "Image " << images;
            cv::namedWindow(winName.str(), cv::WINDOW_AUTOSIZE);
            cv::imshow(winName.str(), out);
        }
    } else {
        std::cerr << "Error opening file" << std::endl;
    }
    cv::waitKey(0);
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

    // Show images
    showImages(g_inputFile, g_number);
    return 0;
}

