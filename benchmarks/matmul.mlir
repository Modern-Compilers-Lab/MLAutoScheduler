

func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
func.func private @printFlops(f64)
func.func private @printI64(i64)
func.func private @printMemrefF32(tensor<*xf32>)

!TTa = tensor<1200x1500xf32>
!TTb = tensor<1500x1000xf32>
!TTc = tensor<1200x1000xf32>
func.func @matmul() -> (!TTc){
  %val = arith.constant 2.00000e+00 : f32

  %input_out = bufferization.alloc_tensor() : !TTa
  %input = linalg.fill ins(%val : f32) outs(%input_out : !TTa) -> !TTa
  %filter_out = bufferization.alloc_tensor() : !TTb
  %filter = linalg.fill ins(%val : f32) outs(%filter_out : !TTb) -> !TTb
  %out = bufferization.alloc_tensor() : !TTc

  %t0 = func.call @nanoTime() : () -> (i64)

  %D = linalg.matmul ins(%input, %filter: !TTa, !TTb)
                   outs(%out: !TTc) -> !TTc

  %t = func.call @nanoTime() : () -> (i64)
  %delta = arith.subi %t, %t0 : i64
  %fp = arith.uitofp %delta : i64 to f64
  func.call @printFlops(%fp) : (f64) -> ()

  return %D : !TTc
}

func.func @main(){

    %c1 = arith.constant 1: index
    %c0 = arith.constant 0 : index
    %n = arith.constant 2: index
    scf.for %i = %c0 to %n step %c1 {
      %outputmain = func.call @matmul() : () -> (!TTc )
    }
 // %unranked = tensor.cast %outputmain : !TTc to tensor<*xf32>
  //func.call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    

    return
}



