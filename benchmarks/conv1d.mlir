

// func.func @conv1d_8_tensor(%input: tensor<11xf32>, %filter: tensor<4xf32>, %output: tensor<8xf32>) -> tensor<8xf32> {
//   %0 = linalg.conv_1d ins(%input, %filter : tensor<11xf32>, tensor<4xf32>)
//                      outs(%output : tensor<8xf32>) -> tensor<8xf32>
//   return %0 : tensor<8xf32>
// }

func.func private @printMemrefF32(tensor<*xf32>)
func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
func.func private @printFlops(f64)
func.func private @printI64(i64)

// Creates and returns a 1-D buffer of size %s1 filled with the value %f
func.func @alloc_1d_filled_f32(%s1 : index, %f : f32) -> tensor<?xf32> {
  %buf = bufferization.alloc_tensor(%s1) : tensor<?xf32>
  %ret = linalg.fill ins(%f : f32) outs(%buf : tensor<?xf32>) -> tensor<?xf32>
  return %ret : tensor<?xf32>
}

func.func @conv_1d(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  %ret = linalg.conv_1d ins (%arg0, %arg1: tensor<?xf32>, tensor<?xf32>)
                outs (%arg2: tensor<?xf32>) -> tensor<?xf32>
  return %ret : tensor<?xf32>
}

func.func @main() {
  %c3 = arith.constant 4 : index
  %c6 = arith.constant 2046 : index
  %c8 = arith.constant 2048 : index
  %f10 = arith.constant 10.00000e+00 : f32
  %val = arith.constant 2.00000e+00 : f32
  %zero = arith.constant 0.00000e+00 : f32
 %t0 = func.call @nanoTime() : () -> (i64)
  %filter1D = call @alloc_1d_filled_f32(%c3, %val) : (index, f32) -> (tensor<?xf32>)
  // %filter = tensor.cast %filter1D : tensor<?xf32> to tensor<4xf32>

  %in1D = call @alloc_1d_filled_f32(%c8, %val) : (index, f32) -> (tensor<?xf32>)
  // %in = tensor.cast %in1D : tensor<?xf32> to tensor<11xf32>

  %out1D = call @alloc_1d_filled_f32(%c6, %zero) : (index, f32) -> (tensor<?xf32>)
  // %out = tensor.cast %out1D : tensor<?xf32> to tensor<8xf32>

 
  %output = call @conv_1d(%in1D, %filter1D, %out1D) : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
  %t1 = func.call @nanoTime() : () -> (i64)
  %delta = arith.subi %t1, %t0 : i64
  %fp = arith.uitofp %delta : i64 to f64
  func.call @printFlops(%fp) : (f64) -> ()
  func.call @printI64(%delta) : (i64) -> ()

  // %unranked = tensor.cast %output : tensor<?xf32> to tensor<*xf32>
  // call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
  bufferization.dealloc_tensor %filter1D : tensor<?xf32>
  // bufferization.dealloc_tensor %filter : tensor<4xf32>
  bufferization.dealloc_tensor %in1D : tensor<?xf32>
  // bufferization.dealloc_tensor %in : tensor<11xf32>
  bufferization.dealloc_tensor %out1D : tensor<?xf32>
  // bufferization.dealloc_tensor %out : tensor<8xf32>
  return
}
