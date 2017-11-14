# MNIST

## Requirements
* Linux/Mac OS
* OpenCV 3.3.1 [GitHub](https://github.com/opencv/opencv/tree/3.3.1)
* C++11
* Build tools: `cd tools/cpp && cmake . && make`

## OCV File Format
*Note: All images are expected to be squares*
images.ocv
```
TOTAL_NUM_OF_IMAGES
LABEL   SQUARE_SIDE     Pixel_11 Pixel_12 ... Pixel_21 ... Pixel_NN
LABEL   SQUARE_SIDE     Pixel_11 Pixel_12 ... Pixel_21 ... Pixel_NN
LABEL   SQUARE_SIDE     Pixel_11 Pixel_12 ... Pixel_21 ... Pixel_NN
LABEL   SQUARE_SIDE     Pixel_11 Pixel_12 ... Pixel_21 ... Pixel_NN
...
```

## Commands

### Prepare and preproecss input
#### Enhance training input format
```
cd tools/bash
./preprocess.sh <input-file> <output-file>
```

#### Generate output directories
```
cd tools/bash
./output-dirs.sh <directory>
```

### Image 2 Images
For all options, use the `--help` flag

#### Display one manipulated image
```
image2images -i <preprocessed-input> \
        -a binary \
        -a clean \
        -a align3 \
        -d
```
Note: To generate PNG images instead of displaying them, replace the `-d` flag with `-o <output-dir>` 
in order to store the files on disk. To store images in separate OCV files, add the `-m` flag. 
To store images in a single OCV file, add the `-M` flag.

#### Display each detected element on a new image
```
image2images -i <preprocessed-input> \
        -a binary \
        -a clean \
        -a split \
        -d
```
Note: To generate PNG images instead of displaying them, replace the `-d` flag with `-o <output-dir>` 
in order to store the files on disk. To store images in separate OCV files, add the `-m` flag. 
To store images in a single OCV file, add the `-M` flag.

### Run kNN
```
image2images -i <preprocessed-input> \
        -k <preprocessed-mnist>
        -a binary \
        -a clean \
        -a split \
        -a validateKNN \
        -n 1000
```
Note: To generate PNG images instead of displaying them, replace the `-d` flag with `-o <output-dir>` 
in order to store the files on disk. To store images in separate OCV files, add the `-m` flag. 
To store images in a single OCV file, add the `-M` flag.

### Run Neural Network
```
./image2images -i <preprocessed-input> \
        -N <preprocessed-input> \
        -a validateNN \
        -n1000
```
Hyperparams `EPOCH`, `BATCH` and `ALPHA` (learning rate) are C++ Macros that can be configured from the file under: `tools/cpp/neuralnetwork/include/neuralnetwork/NeuralNetwork.h`
