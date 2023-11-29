//===- TransformInterpreterPassBase.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base class with shared implementation for transform dialect interpreter
// passes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_TRANSFORMS_TRANSFORMINTERPRETERPASSBASEMODIFIED_H
#define MLIR_DIALECT_TRANSFORM_TRANSFORMS_TRANSFORMINTERPRETERPASSBASEMODIFIED_H

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include <memory>
using namespace mlir;

struct LogicalResult;
class MLIRContext;
class ModuleOp;
class Operation;
template <typename>
class OwningOpRef;
class Region;


/// Template-free implementation of TransformInterpreterPassBase::initialize.
mlir::LogicalResult interpreterBaseInitializeImpl(
    mlir::MLIRContext *context, llvm::StringRef transformFileName,
    llvm::StringRef transformLibraryFileName,
    std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>> &module,
    std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>> &libraryModule,
    function_ref<std::optional<mlir::LogicalResult>(mlir::OpBuilder &, mlir::Location)>
        moduleBuilder = nullptr);
mlir::LogicalResult interpreterBaseInitializeImplModified(
    mlir::MLIRContext *context, llvm::StringRef transformFileName,
    llvm::StringRef transformLibraryFileName,
    std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>> &module,
    std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>> &libraryModule
);
/// Template-free implementation of
/// TransformInterpreterPassBase::runOnOperation.
mlir::LogicalResult interpreterBaseRunOnOperationImpl(
    mlir::Operation *target, llvm::StringRef passName,
    const std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>> &sharedTransformModule,
    const std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>> &libraryModule,
    const RaggedArray<mlir::transform::MappedValue> &extraMappings,
    const mlir::transform::TransformOptions &options,
    const mlir::Pass::Option<std::string> &transformFileName,
    const mlir::Pass::Option<std::string> &transformLibraryFileName,
    const mlir::Pass::Option<std::string> &debugPayloadRootTag,
    const mlir::Pass::Option<std::string> &debugTransformRootTag,
    StringRef binaryName);
mlir::LogicalResult interpreterBaseRunOnOperationImplModified(
    mlir::Operation *target, llvm::StringRef passName,
    const std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>> &sharedTransformModule,
    const std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>> &libraryModule,
    const RaggedArray<mlir::transform::MappedValue> &extraMappings,
    const mlir::transform::TransformOptions &options,
    const mlir::Pass::Option<std::string> &transformFileName,
    const mlir::Pass::Option<std::string> &transformLibraryFileName,
    const mlir::Pass::Option<std::string> &debugPayloadRootTag,
    const mlir::Pass::Option<std::string> &debugTransformRootTag,
    llvm::StringRef binaryName);// namespace detail

/// Base class for transform dialect interpreter passes that can consume and
/// dump transform dialect scripts in separate files. The pass is controlled by
/// three string options:
///
///   - transformFileName: if non-empty, the name of the file containing the
///     transform script. If empty, `debugTransformRootTag` is considered or the
///     pass root operation must contain a single top-level transform op that
///     will be interpreted.
///   - transformLibraryFileName: if non-empty, the name of the file containing
///     definitions of external symbols referenced in the transform script.
///     These definitions will be used to replace declarations.
///   - debugPayloadRootTag: if non-empty, the value of the attribute named
///     `kTransformDialectTagAttrName` indicating the single op that is
///     considered the payload root of the transform interpreter; otherwise, the
///     root operation of the pass is used.
///   - debugTransformRootTag: if non-empty, the value of the attribute named
///     `kTransformDialectTagAttrName` indicating the single top-level transform
///     op contained in the payload root to be used as the entry point by the
///     transform interpreter; mutually exclusive with `transformFileName`.
///
/// The pass runs the transform dialect interpreter as directed by the options.
/// It also provides the mechanism to dump reproducers into stderr
/// (-debug-only=transform-dialect-dump-repro) or into a temporary file
/// (-debug-only=transform-dialect-save-repro) that can be used with this
/// pass in a standalone mode.
///
/// Concrete passes must derive from this class instead of their generated base
/// class (or PassWrapper), and supply themselves and the generated base class
/// as template arguments. They are *not* expected to to implement `initialize`
/// or `runOnOperation`. They *are* expected to call the copy constructor of
/// this class in their copy constructors, short of which the file-based
/// transform dialect script injection facility will become nonoperational.
///
/// Concrete passes may implement the `runBeforeInterpreter` and
/// `runAfterInterpreter` to customize the behavior of the pass.
template <typename Concrete, template <typename> typename GeneratedBase>
class TransformInterpreterPassBaseModified : public GeneratedBase<Concrete> {
public:
  explicit TransformInterpreterPassBaseModified(
      const mlir::transform::TransformOptions &options = mlir::transform::TransformOptions())
      : options(options) {}

  TransformInterpreterPassBaseModified(const TransformInterpreterPassBaseModified &pass) {
    sharedTransformModule = pass.sharedTransformModule;
    options = pass.options;
  }

  static llvm::StringLiteral getBinaryName() { return "mlir-opt"; }

  mlir::LogicalResult initialize(mlir::MLIRContext *context) override {

#define REQUIRE_PASS_OPTION(NAME)                                              \
  static_assert(                                                               \
      std::is_same_v<                                                          \
          std::remove_reference_t<decltype(std::declval<Concrete &>().NAME)>,  \
          mlir::Pass::Option<std::string>>,                                          \
      "required " #NAME " string pass option is missing")

    REQUIRE_PASS_OPTION(transformFileName);
    REQUIRE_PASS_OPTION(debugPayloadRootTag);
    REQUIRE_PASS_OPTION(debugTransformRootTag);
    REQUIRE_PASS_OPTION(transformLibraryFileName);

#undef REQUIRE_PASS_OPTION

    llvm::StringRef transformFileName =
        static_cast<Concrete *>(this)->transformFileName;
    llvm::StringRef transformLibraryFileName =
        static_cast<Concrete *>(this)->transformLibraryFileName;
    return interpreterBaseInitializeImpl(
        context, transformFileName, transformLibraryFileName,
        sharedTransformModule, transformLibraryModule,
        [this](mlir::OpBuilder &builder, mlir::Location loc) {
          return static_cast<Concrete *>(this)->constructTransformModule(
              builder, loc);
        });
  }

  /// Hook for passes to run additional logic in the pass before the
  /// interpreter. If failure is returned, the pass fails and the interpreter is
  /// not run.
  mlir::LogicalResult runBeforeInterpreter(mlir::Operation *) { return mlir::success(); }

  /// Hook for passes to run additional logic in the pass after the interpreter.
  /// Only runs if everything succeeded before. If failure is returned, the pass
  /// fails.
  mlir::LogicalResult runAfterInterpreter(mlir::Operation *) { return mlir::success(); }

  /// Hook for passes to run custom logic to construct the transform module.
  /// This will run during initialization. If the external script is provided,
  /// it overrides the construction, which will not be called.
  std::optional<mlir::LogicalResult> constructTransformModule(mlir::OpBuilder &builder,
                                                        mlir::Location loc) {
    return std::nullopt;
  }

  void runOnOperation() override {
    auto *pass = static_cast<Concrete *>(this);
    mlir::Operation *op = pass->getOperation();
    llvm::StringRef binaryName = Concrete::getBinaryName();
    if (failed(pass->runBeforeInterpreter(op)) ||
        failed(interpreterBaseRunOnOperationImpl(
            op, pass->getArgument(), sharedTransformModule,
            transformLibraryModule,
            /*extraMappings=*/{}, options, pass->transformFileName,
            pass->transformLibraryFileName, pass->debugPayloadRootTag,
            pass->debugTransformRootTag, binaryName)) ||
        failed(pass->runAfterInterpreter(op))) {
      return pass->signalPassFailure();
    }
  }

protected:
  /// Transform interpreter options.
  mlir::transform::TransformOptions options;

  /// Returns a read-only reference to shared transform module.
  const std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>> &
  getSharedTransformModule() const {
    return sharedTransformModule;
  }

  /// Returns a read-only reference to the transform library module.
  const std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>> &
  getTransformLibraryModule() const {
    return transformLibraryModule;
  }


  /// The separate transform module to be used for transformations, shared
  /// across multiple instances of the pass if it is applied in parallel to
  /// avoid potentially expensive cloning. MUST NOT be modified after the pass
  /// has been initialized.
  std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>> sharedTransformModule = nullptr;

  /// The transform module containing symbol definitions that become available
  /// in the transform scripts. Similar to dynamic linking for binaries. This is
  /// shared across multiple instances of the pass and therefore MUST NOT be
  /// modified after the pass has been initialized.
  std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>> transformLibraryModule = nullptr;

  private:
};



#endif // MLIR_DIALECT_TRANSFORM_TRANSFORMS_TRANSFORMINTERPRETERPASSBASEMODIFIED_H
