#include <iostream>
#include <fstream>
#include <unistd.h>
#include <getopt.h>

// Global variables
std::string g_inputFile;

// Codes
const int CODE_ERROR = 1;

/**
 * Print program usage to stdout
 */
void printUsage() {
    std::cout
            << "neuralnetwork - A fully connected feed-forward neural network" << std::endl
            << "    -i, --input      Input file" << std::endl
            << "    -h, --help       Display this help message" << std::endl;
}

/**
 * Initialize parameters
 * @param argc
 * @param argv
 */
void initParams(int argc, char *argv[]) {
    struct option longOptions[] = {
            {"input",   no_argument,       0, 'i'},
            {"help",   no_argument,       0, 'h'},
            {0, 0,                        0, 0}
    };

    int optionIndex = 0;
    int c;
    while ((c = getopt_long(argc, argv, "hi:", longOptions, &optionIndex)) != -1) {
        switch (c) {
            case 'i':
                g_inputFile = optarg;
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

    return 0;
}

