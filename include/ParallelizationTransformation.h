
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
#include "MLIRCodeIR.h"
#include "Node.h"

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
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"



#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <llvm/ADT/SmallVector.h>
// #include "/home/nassimiheb/MLIR/llvm-project/mlir/lib/Dialect/Affine/Transforms/LoopTiling.cpp"

#include <iostream>
#include <random>
#pragma once

class Parallelization: public Transformation{
    private:
        mlir::TilingInterface* op;
        mlir::MLIRContext *context;
        llvm::SmallVector<int64_t, 4> tileSizes;
    public:
        Parallelization();

        /// Constructor for Tiling that allows specifying the tile size.
        Parallelization(mlir::TilingInterface* op, llvm::SmallVector<int64_t, 4> tileSizes, mlir::MLIRContext *context);

        /// Applies the tiling transformation to the given CodeIR object.
        /// Overrides the applyTransformation() method from the base class Transformation.
        void applyTransformation(CodeIR CodeIr) override;
        std::string printTransformation() override;
        /// Creates a list of tiling transformation candidates for the given CodeIR object.
        /// Overrides the createCandidates() method from the base class Transformation.
        static SmallVector<Node* , 2>  createParallelizationCandidates(Node *node, mlir::MLIRContext *context);

        llvm::SmallVector<int64_t, 4>  getTileSizes();
};