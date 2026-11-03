#!/bin/sh
# Builds both benchmarks and runs them at a few thread counts. Apple Silicon (arm64) only.
set -e
mkdir -p build
clang -O2 -o build/stream stream.c -lpthread
clang -O3 -ffast-math -march=armv8.2-a+dotprod -Wall -o build/gemv gemv.c -lpthread
for t in ${THREADS:-1 4 8 12}; do
  ./build/stream "$t"
done
for t in ${THREADS:-1 4 8 12}; do
  ./build/gemv "$t" "${MODEL_PARAMS:-8e9}"
done
