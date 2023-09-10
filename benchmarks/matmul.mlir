func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
func.func private @printFlops(f64)
func.func private @printI64(i64)
func.func private @printMemrefF32(tensor<*xf32>)

func.func @alloc_4d_filled_f32(%s1 : index, %s2 : index, %f : f32) -> tensor<?x?xf32> {
  %buf = bufferization.alloc_tensor(%s1, %s2) : tensor<?x?xf32>
  %ret = linalg.fill ins(%f : f32) outs(%buf : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %ret : tensor<?x?xf32>
}

func.func @mat_mul(%arg0: tensor<64x128xf32>, %arg1: tensor<128x256xf32>, %arg2: tensor<64x256xf32> ) -> tensor<64x256xf32> {

  %ret = linalg.matmul ins(%arg0, %arg1: tensor<64x128xf32>, tensor<128x256xf32>)
                     outs(%arg2: tensor<64x256xf32>) -> tensor<64x256xf32>
  return %ret : tensor<64x256xf32>
}

func.func @main() {

  %c2 = arith.constant 64 : index
  %c3 = arith.constant 128: index
  %c4 = arith.constant 256 : index


  %val = arith.constant 2.00000e+00 : f32
  %zero = arith.constant 0.00000e+00 : f32

  %A_temp = call @alloc_4d_filled_f32(%c2, %c3, %val) :(index, index, f32) -> (tensor<?x?xf32>)
  %A = tensor.cast %A_temp : tensor<?x?xf32> to tensor<64x128xf32>

  %B_temp = call @alloc_4d_filled_f32(%c3, %c4, %val) : (index, index, f32) -> (tensor<?x?xf32>)
  %B = tensor.cast %B_temp : tensor<?x?xf32> to tensor<128x256xf32>

  %C_temp = call @alloc_4d_filled_f32(%c2, %c4, %zero) : (index, index, f32) -> (tensor<?x?xf32>)
  %C = tensor.cast %C_temp : tensor<?x?xf32> to tensor<64x256xf32>


  %t0 = func.call @nanoTime() : () -> (i64)
  %D =  call @mat_mul(%A, %B, %C) : (tensor<64x128xf32>, tensor<128x256xf32>, tensor<64x256xf32>) -> tensor<64x256xf32>
  %t1 = func.call @nanoTime() : () -> (i64)
  %delta = arith.subi %t1, %t0 : i64
  %fp = arith.uitofp %delta : i64 to f64
  func.call @printFlops(%fp) : (f64) -> ()
  func.call @printI64(%delta) : (i64) -> ()


  return
}


