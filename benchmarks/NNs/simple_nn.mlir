#map = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1) -> (0, 0)>

!TTc =  tensor<1x8xf32>
  func.func @__inference_my_predict_1000(%arg0: tensor<1x2xf32>, %arg1: tensor<2x4xf32> , %arg2: tensor<4xf32> , %arg3: tensor<4x8xf32>, %arg4: tensor<8xf32>) -> (!TTc ) {

    %expanded = tensor.expand_shape %arg0 [[0, 1], [2]] : tensor<1x2xf32> into tensor<1x1x2xf32>
    %expanded_0 = tensor.expand_shape %arg1 [[0, 1], [2]] : tensor<2x4xf32> into tensor<1x2x4xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %4 = tensor.empty() : tensor<1x1x4xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
    %6 = linalg.batch_matmul ins(%expanded, %expanded_0 : tensor<1x1x2xf32>, tensor<1x2x4xf32>) outs(%5 : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
    %collapsed = tensor.collapse_shape %6 [[0, 1], [2]] : tensor<1x1x4xf32> into tensor<1x4xf32>
    %expanded_1 = tensor.expand_shape %arg2 [[0, 1]] : tensor<4xf32> into tensor<1x4xf32>
    %7 = tensor.empty() : tensor<1x4xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel"]} ins(%collapsed, %expanded_1 : tensor<1x4xf32>, tensor<1x4xf32>) outs(%7 : tensor<1x4xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %30 = arith.addf %in, %in_11 : f32
      linalg.yield %30 : f32
    } -> tensor<1x4xf32>
    %9 = tensor.empty() : tensor<1x4xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<1x4xf32>) outs(%9 : tensor<1x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_11 = arith.constant 0.000000e+00 : f32
      %cst_12 = arith.constant 3.40282347E+38 : f32
      %30 = arith.minf %in, %cst_12 : f32
      %31 = arith.maxf %30, %cst_11 : f32
      linalg.yield %31 : f32
    } -> tensor<1x4xf32>
    %expanded_2 = tensor.expand_shape %10 [[0, 1], [2]] : tensor<1x4xf32> into tensor<1x1x4xf32>
    %expanded_3 = tensor.expand_shape %arg3 [[0, 1], [2]] : tensor<4x8xf32> into tensor<1x4x8xf32>
    %cst_4 = arith.constant 0.000000e+00 : f32
    %11 = tensor.empty() : tensor<1x1x8xf32>
    %12 = linalg.fill ins(%cst_4 : f32) outs(%11 : tensor<1x1x8xf32>) -> tensor<1x1x8xf32>
    %13 = linalg.batch_matmul ins(%expanded_2, %expanded_3 : tensor<1x1x4xf32>, tensor<1x4x8xf32>) outs(%12 : tensor<1x1x8xf32>) -> tensor<1x1x8xf32>
    %collapsed_5 = tensor.collapse_shape %13 [[0, 1], [2]] : tensor<1x1x8xf32> into tensor<1x8xf32>
    %expanded_6 = tensor.expand_shape %arg4 [[0, 1]] : tensor<8xf32> into tensor<1x8xf32>
    %14 = tensor.empty() : tensor<1x8xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel"]} ins(%collapsed_5, %expanded_6 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%14 : tensor<1x8xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %30 = arith.addf %in, %in_11 : f32
      linalg.yield %30 : f32
    } -> tensor<1x8xf32>
    %16 = tensor.empty() : tensor<1xf32>
    %cst_7 = arith.constant -3.40282347E+38 : f32
    %17 = linalg.fill ins(%cst_7 : f32) outs(%16 : tensor<1xf32>) -> tensor<1xf32>
    %18 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%15 : tensor<1x8xf32>) outs(%17 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %30 = arith.maxf %in, %out : f32
      linalg.yield %30 : f32
    } -> tensor<1xf32>
    %expanded_8 = tensor.expand_shape %18 [[0, 1]] : tensor<1xf32> into tensor<1x1xf32>
    %19 = tensor.empty() : tensor<1x8xf32>
    %20 = linalg.generic {indexing_maps = [#map, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%15, %expanded_8 : tensor<1x8xf32>, tensor<1x1xf32>) outs(%19 : tensor<1x8xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %30 = arith.subf %in, %in_11 : f32
      linalg.yield %30 : f32
    } -> tensor<1x8xf32>
    %21 = tensor.empty() : tensor<1x8xf32>
    %22 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%20 : tensor<1x8xf32>) outs(%21 : tensor<1x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %30 = math.exp %in : f32
      linalg.yield %30 : f32
    } -> tensor<1x8xf32>
    %23 = tensor.empty() : tensor<1xf32>
    %cst_9 = arith.constant 0.000000e+00 : f32
    %24 = linalg.fill ins(%cst_9 : f32) outs(%23 : tensor<1xf32>) -> tensor<1xf32>
    %25 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%22 : tensor<1x8xf32>) outs(%24 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %30 = arith.addf %in, %out : f32
      linalg.yield %30 : f32
    } -> tensor<1xf32>
    %expanded_10 = tensor.expand_shape %25 [[0, 1]] : tensor<1xf32> into tensor<1x1xf32>
    %26 = tensor.empty() : tensor<1x1xf32>
    %27 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_10 : tensor<1x1xf32>) outs(%26 : tensor<1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_11 = arith.constant 1.000000e+00 : f32
      %30 = arith.divf %cst_11, %in : f32
      linalg.yield %30 : f32
    } -> tensor<1x1xf32>
    %28 = tensor.empty() : tensor<1x8xf32>
    %29 = linalg.generic {indexing_maps = [#map, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%22, %27 : tensor<1x8xf32>, tensor<1x1xf32>) outs(%28 : tensor<1x8xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %30 = arith.mulf %in, %in_11 : f32
      linalg.yield %30 : f32
    } -> tensor<1x8xf32>
    return %29 : !TTc
  }


func.func @main(){
    %c1 = arith.constant 1
     : index
    %c0 = arith.constant 0 : index
    %n = arith.constant 2 : index
    scf.for %i = %c0 to %n step %c1 {

    %outputmain = func.call @__inference_my_predict_1000() : () -> !TTc
    }
    //%unranked = tensor.cast %outputmain : !TTc to tensor<*xf32>
    //call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    return
}