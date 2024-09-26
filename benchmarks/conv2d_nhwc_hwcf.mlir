
// func.func private @printMemrefF32(tensor<*xf32>)
// func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
// func.func private @printFlops(f64)
// func.func private @printI64(i64)

// func.func @alloc_4d_filled_f32(%s1 : index, %s2 : index, %s3 : index, %s4 : index, %f : f32) -> tensor<?x?x?x?xf32> {
//   %buf = bufferization.alloc_tensor(%s1, %s2, %s3, %s4) : tensor<?x?x?x?xf32>
//   %ret = linalg.fill ins(%f : f32) outs(%buf : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
//   return %ret : tensor<?x?x?x?xf32>
// }

// func.func @conv_2d_nhwc_hwcf(%arg0: tensor<3x8x8x3xf32>, %arg1: !TTb, %arg2: tensor<3x6x6x1xf32>) -> tensor<3x6x6x1xf32> {
//   %ret = linalg.conv_2d_nhwc_hwcf {dilationsyy = dense<1> : tensor<2xi64>,
//                                      strides = dense<1> : tensor<2xi64>}
//      ins (%arg0, %arg1: tensor<3x8x8x3xf32>, !TTb)
//     outs (%arg2: tensor<3x6x6x1xf32>) -> tensor<3x6x6x1xf32>
//   return %ret : tensor<3x6x6x1xf32>
// }



// func.func @main() {
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %c3 = arith.constant 3 : index
//   %c6 = arith.constant 6 : index
//   %c8 = arith.constant 8 : index
//   %f10 = arith.constant 10.00000e+00 : f32
//   %val = arith.constant 2.00000e+00 : f32
//   %zero = arith.constant 0.00000e+00 : f32

//   %filter2D_nhwc = call @alloc_4d_filled_f32(%c3, %c3, %c3, %c1, %val) :(index, index, index, index, f32) -> (tensor<?x?x?x?xf32>)
//   %in2D_tmp = call @alloc_4d_filled_f32(%c3, %c8, %c8, %c3, %val) : (index, index, index, index, f32) -> (tensor<?x?x?x?xf32>)
//   %in2D_nhwc = tensor.insert %f10 into %in2D_tmp[%c0, %c0, %c3, %c0] : tensor<?x?x?x?xf32>
//   %out2D_nhwc = call @alloc_4d_filled_f32(%c3, %c6, %c6, %c1, %zero) : (index, index, index, index, f32) -> (tensor<?x?x?x?xf32>)

  
//   %t0 = func.call @nanoTime() : () -> (i64)
//   %dense_ret = call @conv_2d_nhwc_hwcf(%in2D_nhwc, %filter2D_nhwc, %out2D_nhwc) : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> (tensor<?x?x?x?xf32>)

//   %t1 = func.call @nanoTime() : () -> (i64)
//   %delta = arith.subi %t1, %t0 : i64
//   %fp = arith.uitofp %delta : i64 to f64
//   func.call @printFlops(%fp) : (f64) -> ()
//   func.call @printI64(%delta) : (i64) -> ()
  
//   %unranked = tensor.cast %dense_ret : tensor<?x?x?x?xf32> to tensor<*xf32>
//   call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()

//   // Free the resources
//   bufferization.dealloc_tensor %in2D_nhwc : tensor<?x?x?x?xf32>
//   bufferization.dealloc_tensor %filter2D_nhwc : tensor<?x?x?x?xf32>
//   bufferization.dealloc_tensor %out2D_nhwc : tensor<?x?x?x?xf32>
//   return
// }


func.func private @printMemrefF32(tensor<*xf32>)
func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
func.func private @printFlops(f64)
func.func private @printI64(i64)

//!TTa = tensor<3x2048x2048x3xf32>
//!TTb = tensor<3x3x3x1xf32>
//!TTc = tensor<3x2046x2046x1xf32>
!TTa = tensor<32x230x230x3xf32>
!TTb = tensor<7x7x3x64xf32>
!TTc = tensor<32x112x112x64xf32>
func.func @conv() -> !TTc {

  %val = arith.constant 2.00000e+00 : f32
  %out = bufferization.alloc_tensor() : !TTa
  %input = linalg.fill ins(%val : f32) outs(%out : !TTa) -> !TTa
  %out1 = bufferization.alloc_tensor() : !TTb
  %filter = linalg.fill ins(%val : f32) outs(%out1 : !TTb) -> !TTb
  %out2 = bufferization.alloc_tensor() : !TTc
  %output = linalg.fill ins(%val : f32) outs(%out2 : !TTc) -> !TTc


  // %1 = shape.shape_of %filter2D_nhwc : tensor<?x?x?x?xf32> -> !shape.shape
  // %2 = shape.with_shape %filter2D_nhwc, %1 : tensor<?x?x?x?xf32>, !shape.shape
  // %3 = shape.value_of %2 :!TTb
 %t0 = func.call @nanoTime() : () -> (i64)
 
  %dense_ret = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                     strides = dense<2> : tensor<2xi64>}
     ins (%input, %filter: !TTa, !TTb )
    outs (%output: !TTc) -> !TTc
 
  %t1 = func.call @nanoTime() : () -> (i64)
  %delta = arith.subi %t1, %t0 : i64
  %fp = arith.uitofp %delta : i64 to f64
  func.call @printFlops(%fp) : (f64) -> ()
  func.call @printI64(%delta) : (i64) -> ()
  
  //%unranked = tensor.cast %dense_ret : !TTc to tensor<*xf32>
  //call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()

  // Free the resources

  return %dense_ret : !TTc 
}
func.func @main(){

   %c1 = arith.constant 1: index
    %c0 = arith.constant 0 : index
    %n = arith.constant 2: index
    scf.for %i = %c0 to %n step %c1 {
    %outputmain = call @conv() : () -> !TTc
    }
    //%unranked = tensor.cast %outputmain : !TTc to tensor<*xf32>
    //call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()


    return
}