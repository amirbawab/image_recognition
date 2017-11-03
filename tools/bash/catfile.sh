#!/bin/bash

printhelp(){
	cat << EOF
Concatenate all ocv files to one csv file
Usage:
	${0} <folder> <new-filename>
EOF
}

main(){
	if [[ ${#@} != 2 ]]; then
		printhelp
	else
		echo $(find $1 -type f -name *.ocv | wc -l) > $2	
		find $1 -type f | xargs -I {} sh -c "cat {}; echo;" >> $2
	fi
	

}

main $@

