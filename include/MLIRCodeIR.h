//===--------------- MLIRCodeIR.h - MLIR code representation --------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the MLIRCodeIR class, which contains the 
/// code representation and its manipulations
///
//===----------------------------------------------------------------------===//
#pragma once

#include "CodeIR.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
using namespace mlir;

// This is a C++ class called MLIRCodeIR that represents the representation of code.
class MLIRCodeIR : public CodeIR{
    private:
        //mlir::OwningOpRef<Operation*>* MLIRIr; // Representation of code.
    public:
        // MLIRCodeIR();

        // /// Constructor that takes a void* parameter and initializes the Ir member variable.
        // MLIRCodeIR(void* Ir);

        /// A function that parses a file and generates the representation of code stored in Ir.
        /// It takes a StringRef parameter called InputFilename and returns an integer to indicate the
        /// status of the parsing process.
        mlir::OwningOpRef<Operation*> parseInputFile(StringRef InputFilename, MLIRContext &context);

        /// Overrides the cloneIr() method from the base class CodeIR.
        /// Returns a pointer to a new instance of MLIRCodeIR.
        CodeIR* cloneIr() override;

        CodeIR* setMLIRIR(Operation* module);
};
