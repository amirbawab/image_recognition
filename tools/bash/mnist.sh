#!/bin/bash

# Print help message
printHelp() {
    cat<<EOF
Preprocess MNIST images
Usage: 
    ${0} <input-file> <x-file> <y-file>
EOF
}

# Starting function
main() {
    if [[ ${#@} != 3 ]]; then
        printHelp
    else
        sed 's/,/ /g' "$1" > "$2"
        cat "$2" | awk '{print$1}' > "$3"
        sed -i -e 's/^\w //g' "$2"
    fi
}

# Start
main $@
