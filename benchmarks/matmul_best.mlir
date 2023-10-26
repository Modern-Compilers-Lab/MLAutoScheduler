//module {
 // func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
 // func.func private @printFlops(f64)
//  func.func private @printI64(i64)
//  func.func private @printMemrefF32(memref<*xf32>)
 // func.func @matmul() -> memref<1200x1000xf32> {
//    %cst = arith.constant 2.000000e+00 : f32
//    %cst_0 = arith.constant 0.000000e+00 : f32
//    %0 = call @nanoTime() : () -> i64
//    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1200x1500xf32>
//    linalg.fill ins(%cst : f32) outs(%alloc : memref<1200x1500xf32>)
//    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1500x1000xf32>
//    linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<1500x1000xf32>)
//    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1200x1000xf32>
//    linalg.fill ins(%cst_0 : f32) outs(%alloc_2 : memref<1200x1000xf32>)
 //   linalg.matmul ins(%alloc, %alloc_1 : memref<1200x1500xf32>, memref<1500x1000xf32>) outs(%alloc_2 : memref<1200x1000xf32>)
//    %1 = call @nanoTime() : () -> i64
//    %2 = arith.subi %1, %0 : i64
//    %3 = arith.uitofp %2 : i64 to f64
//    call @printFlops(%3) : (f64) -> ()
//    call @printI64(%2) : (i64) -> ()
//    memref.dealloc %alloc : memref<1200x1500xf32>
//    memref.dealloc %alloc_1 : memref<1500x1000xf32>
//    return %alloc_2 : memref<1200x1000xf32>
//  }
//  func.func @main() {//
  //  %0 = call @matmul() : () -> memref<1200x1000xf32>
  //  return
//  }
//}
func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
func.func private @printFlops(f64)
func.func private @printI64(i64)
func.func private @printMemrefF32(tensor<*xf32>)

!TTa = tensor<1200x1500xf32>
!TTb = tensor<1500x1000xf32>
!TTc = tensor<1200x1000xf32>


func.func @matmul() -> !TTc{


  %val = arith.constant 2.00000e+00 : f32
  %zero = arith.constant 0.00000e+00 : f32

  %out = bufferization.alloc_tensor() : !TTa
  %A = linalg.fill ins(%val : f32) outs(%out : !TTa) -> !TTa
  %out1 = bufferization.alloc_tensor() : !TTb
  %B = linalg.fill ins(%val : f32) outs(%out1 : !TTb) -> !TTb
  %out2 = bufferization.alloc_tensor() : !TTc
  %C = linalg.fill ins(%zero : f32) outs(%out2 : !TTc) -> !TTc




  %t0 = func.call @nanoTime() : () -> (i64)

  %D = linalg.matmul ins(%A, %B: !TTa, !TTb)
                    outs(%C: !TTc) -> !TTc
  
  %t = func.call @nanoTime() : () -> (i64)
  %delta = arith.subi %t, %t0 : i64
  %fp = arith.uitofp %delta : i64 to f64
  func.call @printFlops(%fp) : (f64) -> ()
  //func.call @printI64(%delta) : (i64) -> ()

  
  
  return %D : !TTc 
}

func.func @main(){
    %c1 = arith.constant 1
     : index
    %c0 = arith.constant 0 : index
    %n = arith.constant 20 : index
    scf.for %i = %c0 to %n step %c1 {

    %outputmain = func.call @matmul() : () -> !TTc
    }
    //%unranked = tensor.cast %outputmain : !TTc to tensor<*xf32>
    //call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    return
}
transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):
  // The original fill op which will be fused into the outer scf.forall created by
  // tiling the convolution.
  %original_fill = transform.structured.match ops{["linalg.fill"]} in %variant_op
    : (!transform.any_op) -> !transform.any_op

  // TODO: Add a transform.structured.specialize that can match a few different ops
  // Then, this reduces to just a linalg.matmul and we can reuse existing strategies.
  %named_conv = transform.structured.match ops{["linalg.matmul"]} in %variant_op
    : (!transform.any_op) -> !transform.any_op

  // Step 1. Tile to forall and sequential scf.for.
  // ======================================================
  %forall_l1, %conv_l1 = transform.structured.tile_to_forall_op %named_conv tile_sizes [10, 10]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  
  transform.structured.fuse_into_containing_op %original_fill into %forall_l1
    : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)


  %1 = transform.structured.generalize %conv_l1 : (!transform.any_op) -> !transform.any_op

  transform.structured.interchange %1 iterator_interchange = [0, 2, 1] : (!transform.any_op) -> !transform.any_op
// Step 3. Vectorize.
  // ======================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op
    : (!transform.any_op) -> !transform.any_op
  %func_0 = transform.structured.vectorize %func {vectorize_padding}
    : (!transform.any_op) -> (!transform.any_op)

  %func_01 = transform.structured.hoist_redundant_vector_transfers %func_0 :
    (!transform.any_op) -> (!transform.any_op)

  transform.structured.hoist_redundant_tensor_subsets %func_01 :
    (!transform.any_op) -> ()

 
}
