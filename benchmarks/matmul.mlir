module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @matmul() -> memref<1200x1000xf32> {
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = call @nanoTime() : () -> i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1200x1500xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<1200x1500xf32>)
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1500x1000xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<1500x1000xf32>)
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1200x1000xf32>
    linalg.fill ins(%cst_0 : f32) outs(%alloc_2 : memref<1200x1000xf32>)
    linalg.matmul ins(%alloc, %alloc_1 : memref<1200x1500xf32>, memref<1500x1000xf32>) outs(%alloc_2 : memref<1200x1000xf32>)
    %1 = call @nanoTime() : () -> i64
    %2 = arith.subi %1, %0 : i64
    %3 = arith.uitofp %2 : i64 to f64
    call @printFlops(%3) : (f64) -> ()
    call @printI64(%2) : (i64) -> ()
    memref.dealloc %alloc : memref<1200x1500xf32>
    memref.dealloc %alloc_1 : memref<1500x1000xf32>
    return %alloc_2 : memref<1200x1000xf32>
  }
  func.func @main() {
    %0 = call @matmul() : () -> memref<1200x1000xf32>
    return
  }
}