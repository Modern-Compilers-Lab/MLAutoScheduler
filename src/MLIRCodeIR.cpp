//===------------------------- MLIRCodeIR.cpp - MLIRCodeIR  -----------------------===//
//
///===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implmentation of the MLIRCodeIR class, which contains the
/// code representation and its manipulations of MLIR dialects.
///
//===----------------------------------------------------------------------===//

#include "MLIRCodeIR.h"

// MLIRCodeIR::MLIRCodeIR(void* Ir){
//     this->Ir = Ir;
// }

mlir::OwningOpRef<Operation *> MLIRCodeIR::parseInputFile(StringRef InputFilename, MLIRContext &context)
{
    // Register any command line options.
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();

    // The input is '.mlir'.
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(InputFilename);
    if (std::error_code ec = fileOrErr.getError())
    {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    }

    // Parse the input mlir.
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    mlir::OwningOpRef<Operation *> module =
        mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module)
    {
        llvm::errs() << "Error can't load file " << InputFilename << "\n";
    }
    // module->dump();
    Operation *newop = (module.get())->clone();
    this->setIr(newop);
    return module;
}

CodeIR *MLIRCodeIR::setMLIRIR( Operation * module)
{
    MLIRCodeIR *clone = new MLIRCodeIR();
    Operation *newop = (module)->clone();
    clone->setIr(newop);
    return clone;
}
CodeIR *MLIRCodeIR::cloneIr()
{
    MLIRCodeIR *clone = new MLIRCodeIR();

    // Create a clone of the Operation object stored in the current MLIRCodeIR's 
    // internal code representation. The cloned Operation is wrapped in an
    // OwningOpRef pointer.
    Operation *op = ((Operation *)(this->getIr()))->clone();
    clone->setIr(op);
    return clone;
}
