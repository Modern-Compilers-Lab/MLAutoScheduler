
func.func private @printMemrefF32(tensor<*xf32>)
func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
func.func private @printFlops(f64)
func.func private @printI64(i64)

!TTa = tensor<256x256x256xf32>
!TTb = tensor<3x3x3xf32>
!TTc = tensor<254x254x254xf32>

func.func @conv_3d()-> !TTc {


  %val = arith.constant 2.00000e+00 : f32
  %zero = arith.constant 0.00000e+00 : f32

  %out = bufferization.alloc_tensor() : !TTa
  %in3D = linalg.fill ins(%val : f32) outs(%out : !TTa) -> !TTa
  %out1 = bufferization.alloc_tensor() : !TTb
  %filter3D = linalg.fill ins(%val : f32) outs(%out1 : !TTb) -> !TTb
  %out2 = bufferization.alloc_tensor() : !TTc
  %out3D = linalg.fill ins(%zero : f32) outs(%out2 : !TTc) -> !TTc


  %t0 = func.call @nanoTime() : () -> (i64)
  %ret_3D = linalg.conv_3d ins (%in3D, %filter3D:!TTa ,!TTb)
                outs (%out3D: !TTc) -> !TTc
   
  
  %t1 = func.call @nanoTime() : () -> (i64)
  %delta = arith.subi %t1, %t0 : i64
  %fp = arith.uitofp %delta : i64 to f64
  func.call @printFlops(%fp) : (f64) -> ()
  func.call @printI64(%delta) : (i64) -> ()

  return %ret_3D : !TTc 
}

func.func @main(){
    %outputmain = call @conv_3d() : () -> !TTc
    //%unranked = tensor.cast %outputmain : !TTc to tensor<*xf32>
    //call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()


    return
}