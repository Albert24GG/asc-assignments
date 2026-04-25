#!/bin/bash

# check that we receive 1 argument which is the name of the valgrind output file
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <valgrind_output_file>"
  exit 1
fi

VALGRIND_OUTPUT_FILE=$1

# replace all occurrences of the path to the source file with just the name of the source file
sed -i 's|/export/home/acs/stud/a/albert.guiman/lab-asc/|/home/albert/src/lab-asc/mnt/|g' "$VALGRIND_OUTPUT_FILE"
