#!/bin/bash

input_string="$1"

# run
/home/nouss/Desktop/PFE/HWNASACOMLIR/Solution/llvm-project/build/bin/mlir-cpu-runner -e main -entry-point-result=void -shared-libs=/home/nouss/Desktop/PFE/HWNASACOMLIR/Solution/llvm-project/build/lib/libmlir_runner_utils.so -shared-libs=/home/nouss/Desktop/PFE/HWNASACOMLIR/Solution/llvm-project/build/lib/libmlir_c_runner_utils.so -shared-libs=/home/nouss/Desktop/PFE/HWNASACOMLIR/MLIR/Autoscheduler/llvm-project/build/lib/libomp.so "./mlir_files/${input_string}_llvm.mlir"

