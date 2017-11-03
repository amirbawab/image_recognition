import argparse
import random
import csv
import os

############## OUTPUT FILE FORMAT ###########
# NUM_OF_LINES                              #
# LABEL 28 P11 P12 P13 ... PNN              #
# LABEL 28 P11 P12 P13 ... PNN              #
# ...                                       #
#############################################

class CLI:
    def read(self):
        """Initialize a command line interface"""

        # Friendly reminder
        print("######### NOTE #########")
        print("This script must only be used for the mnist csv data from:")
        print("https://pjreddie.com/projects/mnist-in-csv/")
        print("########################")

        # Define arguments
        parser = argparse.ArgumentParser(description='Apply algorithms on the csv test file. Header: Id,Text')
        parser.add_argument('-i','--input', nargs=1, help='Input file')
        parser.add_argument('-o','--output', nargs=1, help='Output file')
        args = parser.parse_args()

        # Check for missing arguments
        if args.input is None or args.output is None:
            print("Missing arguments")
            exit(1)

        # Prepare output
        print(">> Generating output file:",args.output[0])
        outputFile = open(args.output[0], 'w')

        # Load input csv
        print(">> Loading input file:", args.input[0])
        with open(args.input[0]) as inFile:
            lines = inFile.read().split("\n")
            outputFile.write("{}\n".format(len(lines)-1))
            for line in lines[0:-1]:
                pixels = line.split(",")
                label = pixels[0]
                outputFile.write("{} {} {}\n".format(label, 28, " ".join(pixels[1:])));

        # Close output files
        outputFile.close()
            
# Stat application
cli = CLI()
cli.read()
