# MNIST

## Requirements
* Linux/Mac OS
* OpenCV 3.3.1 [GitHub](https://github.com/opencv/opencv/tree/3.3.1)
* Compile C++ tools `cd tools/cpp && cmake . && make`

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
        -a detect \
        -a clean \
        -a align \
        -d
```
Note: To generate JPG images instead of displaying them, replace the `-d` flag with `-o <output-dir>` 
in order to store the files on disk. To store images in CSV format, add the `-m` flag.

#### Display permutations of the manipulated images
```
image2images -i <preprocessed-input> \
        -a binary \
        -a detect \
        -a clean \
        -a permutation \
        -d
```
Note: To generate JPG images instead of displaying them, replace the `-d` flag with `-o <output-dir>` 
in order to store the files on disk. To store images in CSV format, add the `-m` flag.

#### Display each detected element on a new image
```
image2images -i <preprocessed-input> \
        -a binary \
        -a detect \
        -a clean \
        -a split \
        -d
```
Note: To generate JPG images instead of displaying them, replace the `-d` flag with `-o <output-dir>` 
in order to store the files on disk. To store images in CSV format, add the `-m` flag.
