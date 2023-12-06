set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name SCF)

add_public_tablegen_target(CustomPassesIncGen)

add_mlir_doc(Passes SCFPasses ./ -gen-pass-doc)
