import os
import struct
import numpy as np

############################################
# THIS SCRIPT IS A MODIFIED VERSION
# OF THE ORIGINAL ONE WHICH CAN BE 
# FOUND HERE:
# https://gist.github.com/akesling/5358964
############################################

### USAGE
# Download MNIST/EMNIST zip file
# Unzip file
# Extract gz files: gunzip *.gz
# Edit this file to point to the files
# Run!

def read(path = "."):

    fname_img = os.path.join(path, 'emnist-letters-train-images-idx3-ubyte')
    fname_lbl = os.path.join(path, 'emnist-letters-train-labels-idx1-ubyte')

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)

def ascii_show(image):
    for y in image:
         row = ""
         for x in y:
             row += '{0: <4}'.format(x)
         print row

def writeLine(ocvFile, item):
    label = item[0]
    if label == 1:
        label = 10
    else:
        label = 11
    ocvFile.write("{} {}".format(label, 28))
    for y in item[1]:
        for x in y:
            ocvFile.write(" {}".format(x))


def ascii_show(image):
    for y in image:
         row = ""
         for x in y:
             row += '{0: <4}'.format(x)
         print row

# Start
ocvFile = open('emnist.ocv', 'w')
mnistList = list(read())
total = 0
for item in mnistList:
    # `a` and `m` only
    if item[0] == 1 or item[0] == 13:
        total += 1

ocvFile.write("{}\n".format(total))
for item in mnistList:
    if item[0] == 1 or item[0] == 13:
        writeLine(ocvFile, item)
        #ascii_show(item[1])
        ocvFile.write("\n")
ocvFile.close()
