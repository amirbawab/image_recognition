#pragma once

#define BATCH 1
#define EPOCH 30
#define ALPHA 0.01
#define VERBOSE false

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

    void train(std::vector<std::vector<int>> X, std::vector<std::vector<int>> Y) {

        // Apply the learning several times
        for(int epoch=0; epoch < EPOCH; epoch++) {
            std::cout << "Doing epoch " << epoch << std::endl;

            // Reset all delta
            resetDelta();

            // Start from first example
            int example = 0;
            while(example < X.size()) {

                // Update weights at the end of each batch size
                if(VERBOSE) {
                    std::cout << "Starting a new batch of size " << BATCH << " with example " << example << std::endl;
                }

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
                    if(VERBOSE) {
                        std::cout << ">> Forward propagation" << std::endl;
                    }
                    for(size_t layer = 1; layer < layers.size(); layer++) {
                        if(VERBOSE) {
                            std::cout << ">>>> For layer " << layer << std::endl;
                        }
                        for(int nId =1; nId < layers[layer]->nodes.size(); nId++) {
                            if(VERBOSE)
                                std::cout << ">>>>>> For node " << nId << std::endl;
                            std::shared_ptr<Node> curNode = layers[layer]->nodes[nId];
                            curNode->z = 0;
                            if(VERBOSE) {
                                std::cout << "g(z)=g(";
                            }
                            for(int inId = 0; inId < curNode->inEdges.size(); inId++) {
                                curNode->z += curNode->inEdges[inId]->weight * curNode->inEdges[inId]->fromNode->z;
                                if(VERBOSE) {
                                    if(inId > 0) {
                                        std::cout << " + ";
                                    }
                                    std::cout << curNode->inEdges[inId]->weight << " * " << curNode->inEdges[inId]->fromNode->z;
                                }
                            }
                            if(VERBOSE) {
                                std::cout << ") = g(" << curNode->z << ") = ";
                            }
                            curNode->sigmoidZ();
                            if(VERBOSE) {
                                std::cout << curNode->z << std::endl;
                            }
                        }
                    }

                    // Back propagation
                    // Step 1: Compute the delta for the output layer
                    if(VERBOSE) {
                        std::cout << std::endl << ">> Back propagation" << std::endl;
                        std::cout << ">>>> For output layer" << std::endl;
                    }
                    for(size_t nId =1; nId < layers.back()->nodes.size(); nId++) {
                        if(VERBOSE) {
                            std::cout << ">>>>>> For node " << nId << std::endl;
                        }
                        std::shared_ptr<Node> outNode = layers.back()->nodes[nId];
                        outNode->delta = outNode->z * (1.0 -outNode->z) * (outNode->z - Y[example][nId-1]);
                        if(VERBOSE) {
                            std::cout << "Delta_out=" << outNode->z << " * (1 - " << outNode->z << ") * (" << outNode->z << " - " << Y[example][nId-1] << ") = " << outNode->delta << std::endl;
                        }
                    }

                    // Step 2: Compute the delta for the hidden layers
                    for(size_t layer = layers.size()-2; layer > 0; layer--) {
                        if(VERBOSE) {
                            std::cout << std::endl << ">> For hidden layer " << layer << std::endl;
                        }
                        for(int nId =1; nId < layers[layer]->nodes.size(); nId++) {
                            if(VERBOSE) {
                                std::cout << ">>>> For node " << nId << std::endl;
                            }
                            std::shared_ptr<Node> hidNode = layers[layer]->nodes[nId];
                            double delta1 = hidNode->z*(1.0 - hidNode->z);
                            double delta2 = 0;
                            if(VERBOSE) {
                                std::cout << ">>>>>> Delta_hid=" << hidNode->z << " * (1 - " << hidNode->z << ")";
                            }
                            for(int outId =0; outId < hidNode->outEdges.size(); outId++) {
                                delta2 += hidNode->outEdges[outId]->weight * hidNode->outEdges[outId]->toNode->delta;
                                if(VERBOSE) {
                                    std::cout << " + " << hidNode->outEdges[outId]->weight << " * " << hidNode->outEdges[outId]->toNode->delta;
                                }
                            }
                            hidNode->delta += delta1 * delta2;
                            if(VERBOSE) {
                                std::cout << " = " << hidNode->delta << std::endl;
                            }
                        }
                    }

                    // Move to next example
                    ++example;
                }

                // Update weights
                if(VERBOSE) {
                    std::cout << std::endl << ">> Updating weights" << std::endl;
                }
                for(int layer=0; layer < layers.size()-1; layer++) {
                    if(VERBOSE) {
                        std::cout << ">>>> For layer " << layer << std::endl;
                    }
                    std::shared_ptr<Layer> curLayer = layers[layer];
                    for(int node=0; node < curLayer->nodes.size(); node++) {
                        if(VERBOSE) {
                            std::cout << ">>>>>> For node " << node << std::endl;
                        }
                        std::shared_ptr<Node> curNode = curLayer->nodes[node];
                        for(int edge=0; edge < curNode->outEdges.size(); edge++) {
                            if(VERBOSE) {
                                std::cout << ">>>>>>>> For edge " << edge << std::endl;
                            }
                            std::shared_ptr<Edge> curEdge = curNode->outEdges[edge];
                            if(VERBOSE) {
                                std::cout << "weight = " << curEdge->weight << " - " << ALPHA << " * (" << curEdge->toNode->delta/BATCH << ") * " << curEdge->fromNode->z;
                            }
                            curEdge->weight -= ALPHA * (curEdge->toNode->delta/BATCH) * curEdge->fromNode->z;
                            if(VERBOSE) {
                                std::cout << " = " << curEdge->weight << std::endl;
                            }
                        }
                    }
                }
                if(VERBOSE) {
                    std::cout << "===============================" << std::endl;
                }
            }
            if(VERBOSE) {
                std::cout << "============================================================================" << std::endl;
            }
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
