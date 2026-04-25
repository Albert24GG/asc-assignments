#!/bin/bash

TOL="0.000001"

./compare out1 ../references/out1 "$TOL"
./compare out2 ../references/out2 "$TOL"
./compare out3 ../references/out3 "$TOL"
