//===- TransformDialectInterpreter.h --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a pass that interprets Transform dialect operations in
// the module.
//
//===----------------------------------------------------------------------===//

using namespace mlir; 

std::unique_ptr<mlir::Pass>
createTransformDialectInterpreterPass(llvm::StringRef transformOp);

/*auto
createTransformDialectEraseSchedulePass();*/

/*using namespace mlir;

namespace {
std::unique_ptr<mlir::Pass>
createTransformDialectInterpreterPass(llvm::StringRef transformOp);
}*/