#!/bin/bash

# TODO: replace this with your implementation
#       it doesn't have to be a bash script.
#       It can be a python script or even a
#       compiled executable.

# Usage: ./my_solver.sh [sat_solver] [path_to_task]

SATSOLVER=$1
TASK=$2

# Simulate work
sleep .1
>&2 echo "running $0 $*"
filename="$(basename $2)"
echo $filename
#if [ $filename == "bsp-sudoku1.txt" ]; then
#    cat $(dirname $2)/$(basename -s .txt $2).sol
#fi
$SATSOLVER $TASK > 6x6.txt

>&2 echo "done!"
