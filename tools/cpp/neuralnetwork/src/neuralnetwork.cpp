#include <iostream>
#include <fstream>
#include <unistd.h>
#include <memory>
#include <vector>
#include <cmath>
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

/*****************
 * NEURAL NETWORK
 ****************/

struct Node;
struct Edge {
    double weight = 0;
    std::shared_ptr<Node> fromNode;
    std::shared_ptr<Node> toNode;
};

struct Node : public std::enable_shared_from_this {
    double _z = 0;
    std::vector<std::shared_ptr<Edge>> inEdges;
    std::vector<std::shared_ptr<Edge>> outEdges;
    void connectTo(std::shared_ptr<Node> node) {
        std::shared_ptr<Edge> newEdge = std::make_shared<Edge>();
        newEdge->weight = (double) rand() / (RAND_MAX + 1.0);
        newEdge->fromNode = shared_from_this();
        newEdge->toNode = node;
        node->inEdges.push_back(newEdge);
        outEdges.push_back(newEdge);
    }
    double sigmoidZ() {
        return 1.0d / (1.0d + std::exp(-_z));
    }
};

struct Layer {
    std::vector<std::shared_ptr<Node>> nodes;
    explicit Layer(int num) {

        // Create num+1 nodes
        for(int i=0; i <= num; i++) {
            std::shared_ptr<Node> newNode = std::make_shared<Node>();
            nodes.push_back(newNode);
        }

        // Configure bias node
        if(!nodes.empty()) {
            nodes.front()->_z = 1;
        }
    }
};

struct Network {
    std::vector<std::shared_ptr<Layer>> layers;
    void addLayer(int nodes) {
        std::shared_ptr<Layer> newLayer = std::make_shared<Layer>(nodes);

        // If not input layer
        if(!layers.empty()) {
            for(int i =1 /*skip bias*/; i < layers.back()->nodes.size(); i++) {
                for(int j=0; j < newLayer->nodes.size(); j++) {
                    layers.back()->nodes[i]->connectTo(newLayer->nodes[j]);
                }
            }
        }
        layers.push_back(newLayer);
    }
};

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


    return 0;
}

