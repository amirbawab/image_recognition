#include <iostream>
#include <fstream>
#include <unistd.h>
#include <memory>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <getopt.h>
#include <neuralnetwork/NeuralNetwork.h>

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
    // TODO Uncomment this later
//    if(g_inputFile.empty()) {
//        printUsage();
//        return 0;
//    }

    // Configure seeded random
    srand(time(NULL));

    // Small network
    struct Network andNet;
    andNet.addLayer(2);
    andNet.addLayer(2);
    andNet.addLayer(1);
//    andNet.layers[0]->nodes[0]->outEdges[0]->weight = 0.8;
//    andNet.layers[0]->nodes[0]->outEdges[1]->weight = -0.1;
//    andNet.layers[0]->nodes[1]->outEdges[0]->weight = 0.5;
//    andNet.layers[0]->nodes[1]->outEdges[1]->weight = 0.9;
//    andNet.layers[0]->nodes[2]->outEdges[0]->weight = 0.4;
//    andNet.layers[0]->nodes[2]->outEdges[1]->weight = 1.0;
//    andNet.layers[1]->nodes[0]->outEdges[0]->weight = 0.3;
//    andNet.layers[1]->nodes[1]->outEdges[0]->weight = -1.2;
//    andNet.layers[1]->nodes[2]->outEdges[0]->weight = 1.1;
    std::vector<std::vector<int>> X = {{1,1},{0,0},{1,0},{0,1}};
    std::vector<std::vector<int>> Y = {{1},{1},{0},{0}};
    andNet.train(X, Y);

    // Predict for the same input
    for(int i=0; i < X.size(); i++) {
        std::cout << "Predicting for example " << i << ": ";
        for(double out : andNet.predict(X[i])) {
            std::cout << out << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

