#!/bin/bash

input_string="$1"


# bufferizing ml_program
#/home/nouss/Desktop/PFE/torch-mlir/build/bin/torch-mlir-opt -refback-mlprogram-bufferize "/home/nouss/Desktop/PFE/HWNASACOMLIR/Solution/CONAS/search_space/mlir_files/${input_string}.mlir" -o "/home/nouss/Desktop/PFE/HWNASACOMLIR/Solution/CONAS/search_space/mlir_files/${input_string}.mlir"

# wrapping call @forward in a main function
python ./touchup.py ${input_string}
python ./wrap.py ${input_string}


# bufferizing func, arith, tensor, linalg
/home/nouss/Desktop/PFE/HWNASACOMLIR/MLIR/Autoscheduler/llvm-project/build/bin/mlir-opt "./mlir_files/${input_string}.mlir" -func-bufferize -linalg-bufferize -arith-bufferize --empty-tensor-to-alloc-tensor  -tensor-bufferize -o "./mlir_files/${input_string}_buff.mlir"


# lowerings
/home/nouss/Desktop/PFE/HWNASACOMLIR/Solution/llvm-project/build/bin/mlir-opt "./mlir_files/${input_string}.mlir"  -loop-invariant-code-motion -cse -canonicalize -cse -eliminate-empty-tensors -empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -convert-linalg-to-loops  -convert-vector-to-scf -convert-scf-to-openmp -canonicalize -lower-affine -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-arith-to-llvm -convert-math-to-llvm -convert-openmp-to-llvm -convert-math-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  -o "./mlir_files/${input_string}_llvm.mlir"


