import argparse
import random
import csv
import os

############# INPUT FILE FORMAT #############
# P11,P12,P13,...,PNN                      #
# P11,P12,P13,...,PNN                      #
# ...                                       #
#############################################

############## LABEL FILE FORMAT ############
# LABEL                                     #
# LABEL                                     #
# ...                                       #
#############################################

############## OUTPUT FILE FORMAT ###########
# NUM_OF_LINES                              #
# LABEL|777 SQUARE_SIDE P11 P12 P13 ... PNN #
# LABEL|777 SQUARE_SIDE P11 P12 P13 ... PNN #
# ...                                       #
#############################################

class CLI:
    def read(self):
        """Initialize a command line interface"""

        # Friendly reminder
        print("######### NOTE #########")
        print("Just remember that the image is expected to be a square")
        print("########################")

        # Define arguments
        parser = argparse.ArgumentParser(description='Apply algorithms on the csv test file. Header: Id,Text')
        parser.add_argument('-i','--input', nargs=1, help='Input file')
        parser.add_argument('-l','--label', nargs=1, help='Label file')
        parser.add_argument('-o','--output', nargs=1, help='Output file')
        args = parser.parse_args()

        # Check for missing arguments
        if args.input is None or args.output is None:
            print("Missing arguments")
            exit(1)

        # Prepare output
        print(">> Generating output file:", args.output[0])
        outputFile = open(args.output[0], 'w')

        # Check if label file attached
        labels = []
        if args.label is not None:
            print(">> Loading labels:", args.label[0])
            with open(args.label[0]) as lFile:
                labels = lFile.read().split("\n")

        # Load input csv
        print(">> Loading input file:", args.input[0])
        lineCount = 0
        with open(args.input[0]) as inFile:
            lines = inFile.read().split("\n")
            outputFile.write("{}\n".format(len(lines)-1))
            for line in lines:
                pixels = line.split(",")
                side = int(len(pixels)**(0.5))
                if side > 1:
                    label = 777
                    if lineCount < len(labels):
                        label = labels[lineCount]
                    outputFile.write("{} {} {}\n".format(label, side, " ".join(pixels)));
                    lineCount+=1

        # Close output file
        outputFile.close()
            
# Stat application
cli = CLI()
cli.read()
