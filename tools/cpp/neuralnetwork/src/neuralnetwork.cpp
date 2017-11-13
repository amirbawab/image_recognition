#include <iostream>
#include <fstream>
#include <unistd.h>
#include <memory>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
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

#define BATCH 1
#define EPOCH 100000
#define ALPHA 0.000001

struct Node;
struct Edge {
    double weight = 0;
    std::shared_ptr<Node> fromNode;
    std::shared_ptr<Node> toNode;
};

struct Node : public std::enable_shared_from_this<Node> {
    double z = 0;
    double delta = 0;
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
    void sigmoidZ() {
        z = 1.0 / (1.0 + std::exp(-z));
    }
    std::string str() {
        std::stringstream ss;
        ss << "[w:";
        for(int i=0; i < inEdges.size(); i++) {
            if(i > 0) {
                ss << ", ";
            }
            ss << inEdges[i]->weight;
        }
        ss << "](v:";
        ss << z << ")";
        return ss.str();
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
            nodes.front()->z = 1;
        }
    }
    std::string str() {
        std::stringstream ss;
        for(int i=0; i < nodes.size(); i++) {
            if(nodes[i]->inEdges.size() > 0 || nodes[i]->outEdges.size() > 0) {
                ss << nodes[i]->str() << " ";
            }
        }
        return ss.str();
    }
};

struct Network {
    std::vector<std::shared_ptr<Layer>> layers;
    void addLayer(int nodes) {
        std::shared_ptr<Layer> newLayer = std::make_shared<Layer>(nodes);

        // If not input layer
        if(!layers.empty()) {
            for(int i =0; i < layers.back()->nodes.size(); i++) {
                for(int j=1; j < newLayer->nodes.size(); j++) {
                    layers.back()->nodes[i]->connectTo(newLayer->nodes[j]);
                }
            }
        }
        layers.push_back(newLayer);
    }

    std::vector<double> predict(std::vector<int> X) {
        std::vector<double> output;

        // Update the input nodes values
        for(int i=1; i < layers.front()->nodes.size(); i++) {
            layers.front()->nodes[i]->z = X[i-1];
        }

        // Feed forward
        for(size_t layer = 1; layer < layers.size(); layer++) {
            for(int nId =1; nId < layers[layer]->nodes.size(); nId++) {
                std::shared_ptr<Node> curNode = layers[layer]->nodes[nId];
                curNode->z = 0;
                for(int inId = 0; inId < curNode->inEdges.size(); inId++) {
                    curNode->z += curNode->inEdges[inId]->weight * curNode->inEdges[inId]->fromNode->z;
                }
                curNode->sigmoidZ();
            }
        }

        // Copy results from output nodes
        for(size_t node=1; node < layers.back()->nodes.size(); node++) {
            output.push_back(layers.back()->nodes[node]->z);
        }
        return output;
    }

    void resetDelta() {
        for(int layer=0; layer < layers.size()-1; layer++) {
            std::shared_ptr<Layer> curLayer = layers[layer];
            for (int node = 0; node < curLayer->nodes.size(); node++) {
                std::shared_ptr<Node> curNode = curLayer->nodes[node];
                curNode->delta = 0;
            }
        }
    }

    void run(std::vector<std::vector<int>> X, std::vector<std::vector<int>> Y) {

        // Apply the learning several times
        for(int epoch=0; epoch < EPOCH; epoch++) {
            std::cout << "Doing epoch " << epoch << std::endl;

            // Reset all delta
            resetDelta();

            // Start from first example
            int example = 0;
            while(example < X.size()) {

                // Update weights at the end of each batch size
                std::cout << "Starting a new batch of size " << BATCH << " with example " << example << std::endl;
                for(int batch = 0; batch < BATCH && example < X.size(); batch++) {

                    // Check that the input and output does not violate the network architecture
                    if(layers.size() < 2
                       || layers.front()->nodes.size()-1 != X[example].size()
                       || layers.back()->nodes.size()-1 != Y[example].size()) {
                        std::cerr << "Example #" << example << " violates the network architecture: skipping ..." << std::endl;
                        continue;
                    }

                    // Update the input nodes values
                    for(int i=1; i < layers.front()->nodes.size(); i++) {
                        layers.front()->nodes[i]->z = X[example][i-1];
                    }

                    // Feed forward
                    std::cout << ">> Forward propagation" << std::endl;
                    for(size_t layer = 1; layer < layers.size(); layer++) {
                        std::cout << ">>>> For layer " << layer << std::endl;
                        for(int nId =1; nId < layers[layer]->nodes.size(); nId++) {
                            std::cout << ">>>>>> For node " << nId << std::endl;
                            std::shared_ptr<Node> curNode = layers[layer]->nodes[nId];
                            curNode->z = 0;
                            std::cout << "g(z)=g(";
                            for(int inId = 0; inId < curNode->inEdges.size(); inId++) {
                                curNode->z += curNode->inEdges[inId]->weight * curNode->inEdges[inId]->fromNode->z;
                                if(inId > 0) {
                                    std::cout << " + ";
                                }
                                std::cout << curNode->inEdges[inId]->weight << " * " << curNode->inEdges[inId]->fromNode->z;
                            }
                            std::cout << ") = g(" << curNode->z << ") = ";
                            curNode->sigmoidZ();
                            std::cout << curNode->z << std::endl;
                        }
                    }

                    // Back propagation
                    // Step 1: Compute the delta for the output layer
                    std::cout << std::endl << ">> Back propagation" << std::endl;
                    std::cout << ">>>> For output layer" << std::endl;
                    for(size_t nId =1; nId < layers.back()->nodes.size(); nId++) {
                        std::cout << ">>>>>> For node " << nId << std::endl;
                        std::shared_ptr<Node> outNode = layers.back()->nodes[nId];
                        outNode->delta = outNode->z * (1.0 -outNode->z) * (outNode->z - Y[example][nId-1]);
                        std::cout << "Delta_out=" << outNode->z << " * (1 - " << outNode->z << ") * (" << outNode->z << " - " << Y[example][nId-1] << ") = " << outNode->delta << std::endl;
                    }

                    // Step 2: Compute the delta for the hidden layers
                    for(size_t layer = layers.size()-2; layer > 0; layer--) {
                        std::cout << std::endl << ">> For hidden layer " << layer << std::endl;
                        for(int nId =1; nId < layers[layer]->nodes.size(); nId++) {
                            std::cout << ">>>> For node " << nId << std::endl;
                            std::shared_ptr<Node> hidNode = layers[layer]->nodes[nId];
                            double delta1 = hidNode->z*(1.0 - hidNode->z);
                            double delta2 = 0;
                            std::cout << ">>>>>> Delta_hid=" << hidNode->z << " * (1 - " << hidNode->z << ")";
                            for(int outId =0; outId < hidNode->outEdges.size(); outId++) {
                                delta2 += hidNode->outEdges[outId]->weight * hidNode->outEdges[outId]->toNode->delta;
                                std::cout << " + " << hidNode->outEdges[outId]->weight << " * " << hidNode->outEdges[outId]->toNode->delta;
                            }
                            hidNode->delta += delta1 * delta2;
                            std::cout << " = " << hidNode->delta << std::endl;
                        }
                    }

                    // Move to next example
                    ++example;
                }

                // Update weights
                std::cout << std::endl << ">> Updating weights" << std::endl;
                for(int layer=0; layer < layers.size()-1; layer++) {
                    std::cout << ">>>> For layer " << layer << std::endl;
                    std::shared_ptr<Layer> curLayer = layers[layer];
                    for(int node=0; node < curLayer->nodes.size(); node++) {
                        std::cout << ">>>>>> For node " << node << std::endl;
                        std::shared_ptr<Node> curNode = curLayer->nodes[node];
                        for(int edge=0; edge < curNode->outEdges.size(); edge++) {
                            std::cout << ">>>>>>>> For edge " << edge << std::endl;
                            std::shared_ptr<Edge> curEdge = curNode->outEdges[edge];
                            std::cout << "weight = " << curEdge->weight << " - " << ALPHA << " * (" << curEdge->toNode->delta/BATCH << ") * " << curEdge->fromNode->z;
                            curEdge->weight -= ALPHA * (curEdge->toNode->delta/BATCH) * curEdge->fromNode->z;
                            std::cout << " = " << curEdge->weight << std::endl;
                        }
                    }
                }
                std::cout << "===============================" << std::endl;
            }
            std::cout << "============================================================================" << std::endl;
        }
    }
    std::string str() {
        std::stringstream ss;
        for(int i=0; i < layers.size(); i++) {
            ss << "Layer " << i+1 << std::endl;
            ss << layers[i]->str() << std::endl << std::endl;
        }
        return ss.str();
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
    andNet.run(X, Y);

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

