
//===----------------------- TilingTransformation.h ----------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the TilingTransformation class, which  
/// contains the declartion of the tiling transformation
///
//===----------------------------------------------------------------------===//

#include "Transformation.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"


#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"

#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Affine/LoopUtils.h"


#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Dialect/Affine/Analysis/Utils.h"
// #include "/home/nassimiheb/MLIR/llvm-project/mlir/lib/Dialect/Affine/Transforms/LoopTiling.cpp"

#include <string>
#include <iostream>
#pragma once

class Tiling: public Transformation{
    private:
        int tileSize; 
    public:
        Tiling();

        /// Constructor for Tiling that allows specifying the tile size.
        Tiling(int tileSize);

        /// Applies the tiling transformation to the given CodeIR object.
        /// Overrides the applyTransformation() method from the base class Transformation.
        void applyTransformation(CodeIR CodeIr) override;

        /// Creates a list of tiling transformation candidates for the given CodeIR object.
        /// Overrides the createCandidates() method from the base class Transformation.
        std::list<Transformation*> createCandidates(CodeIR* CodeIr) override;
};