1. Finding the best parameters for tiling : Testing whether algorithm works on Tiling (which tilesize) - 2D & 3D - with benchmarks and actual code. 
2. Finding the best unrolling factor. 


3. Add the BOUNDS


 %1 = call @alloc_4d_filled_f32(%c128, %c256, %cst) 


 ignore:
  %1 = linalg.fill ins(%arg2 : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %1 : tensor<?x?xf32>


work on :
 %1 = linalg.matmul ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<128x256xf32>) outs(%arg2 : tensor<64x256xf32>) -> tensor<64x256xf32>
    return %1 : tensor<64x256xf32>


ARE WE GETTING BETTER RESULTS CONSISTENTLY. 
run multiple bench marks - understands how evaluation time change. 

1. run  4 benchmarks 
2. 2D variation
3. Different loop sizes -whether it makes a difference (does it always stop before Max)
4. Store different json. Format