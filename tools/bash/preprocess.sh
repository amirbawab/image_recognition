#!/bin/bash

# Print help message
printHelp() {
    cat<<EOF
Preprocess images
Usage: 
    ${0} <input-file> <output-file>
EOF
}

# Starting function
main() {
    if [[ ${#@} != 2 ]]; then
        printHelp
    else
        sed 's/,/ /g' $1 > $2
    fi
}

# Start
main $@
