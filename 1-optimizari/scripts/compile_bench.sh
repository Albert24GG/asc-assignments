#!/bin/bash

# check that we receive 2 arguments which are the name of the source file and the name of the output file
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <source_file.c> <output_file>"
  exit 1
fi

SOURCE_FILE=$1
OUTPUT_FILE=$2

gcc -g -I ~/lab-asc/asc-assignments/1-optimizari/src/ -Wall -Werror -O0 -march=native -lm -o "$OUTPUT_FILE" "$SOURCE_FILE" ~/lab-asc/asc-assignments/1-optimizari/src/{benchmark,argparse}.c
