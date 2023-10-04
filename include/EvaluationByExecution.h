//===----------------------- EvaluationByExecution.h ----------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the EvaluationByExecution class, which  
/// contains an evaluator of the transformed code, the evaluator returns a the 
/// the reel execution time after running the code
///
//===----------------------------------------------------------------------===//

#include "Evaluation.h"
#include "Node.h"

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/BuiltinOps.h"

#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"


#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

#include "mlir/Target/LLVMIR/Dialect/All.h"

#include <utility>
#include <chrono>
#include <iostream>

#include <stdio.h>
#include<sys/wait.h>
#include<unistd.h>


#define READ 0
#define WRITE 1


#pragma once

using namespace mlir;
class EvaluationByExecution {
    public:
        /// Evaluates the transformation by executing it with the given parameters.
        /// Parameters:
        /// - registry: A reference to the DialectRegistry used for execution.
        /// - node: A pointer to the Node object representing the transformation.
        /// Returns: The evaluation result as a double value.
        std::string evaluateTransformation(/*int argc, char** argv, DialectRegistry &registry,*/ Node* node);
};