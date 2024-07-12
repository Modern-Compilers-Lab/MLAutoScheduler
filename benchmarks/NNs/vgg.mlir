#map = affine_map<(d0, d1, d2, d3) -> (d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1, d0)>
#map5 = affine_map<(d0, d1) -> (0, d1)>
#map6 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "VGG11"} {
  memref.global "private" @global_seed : memref<i64> = dense<0>
  func.func @forward() -> tensor<1x1000xf32> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = bufferization.alloc_tensor() : tensor<1x3x224x224xf32>
    %cst_2 = bufferization.alloc_tensor() : tensor<16x3x3x3xf32>
    %cst_3 = bufferization.alloc_tensor() : tensor<16xf32>
    %cst_4 = bufferization.alloc_tensor() : tensor<32x16x3x3xf32>
    %cst_5 = bufferization.alloc_tensor() : tensor<32xf32>
    %cst_6 = bufferization.alloc_tensor() : tensor<64x32x3x3xf32>
    %cst_7 = bufferization.alloc_tensor() : tensor<64xf32>
    %cst_8 = bufferization.alloc_tensor() : tensor<64x64x3x3xf32>
    %cst_9 = bufferization.alloc_tensor() : tensor<64xf32>
    %cst_10 = bufferization.alloc_tensor() : tensor<128x64x3x3xf32>
    %cst_11 = bufferization.alloc_tensor() : tensor<128xf32>
    %cst_12 = bufferization.alloc_tensor() : tensor<128x128x3x3xf32>
    %cst_13 = bufferization.alloc_tensor() : tensor<128xf32>
    %cst_14 = bufferization.alloc_tensor() : tensor<128x128x3x3xf32>
    %cst_15 = bufferization.alloc_tensor() :  tensor<128xf32>
    %cst_16 = bufferization.alloc_tensor() : tensor<128x128x3x3xf32>
    %cst_17 = bufferization.alloc_tensor() : tensor<128xf32>
    %cst_18 = bufferization.alloc_tensor() : tensor<1000x6272xf32>
    %cst_19 = bufferization.alloc_tensor() : tensor<1000xf32>
    %cst_20 = bufferization.alloc_tensor() : tensor<1000x1000xf32>
    %cst_21 = bufferization.alloc_tensor() : tensor<1000xf32>
    %cst_22 = bufferization.alloc_tensor() : tensor<1000x1000xf32>
    %cst_23 = bufferization.alloc_tensor() : tensor<1000xf32>

    %padded = tensor.pad %cst_1 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x3x224x224xf32> to tensor<1x3x226x226xf32>
    %0 = tensor.empty() : tensor<1x16x224x224xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3 : tensor<16xf32>) outs(%0 : tensor<1x16x224x224xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x16x224x224xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded, %cst_2 : tensor<1x3x226x226xf32>, tensor<16x3x3x3xf32>) outs(%1 : tensor<1x16x224x224xf32>) -> tensor<1x16x224x224xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<1x16x224x224xf32>) outs(%0 : tensor<1x16x224x224xf32>) {
    ^bb0(%in: f32, %out: f32):
      %62 = arith.cmpf ugt, %in, %cst_0 : f32
      %63 = arith.select %62, %in, %cst_0 : f32
      linalg.yield %63 : f32
    } -> tensor<1x16x224x224xf32>
    %4 = tensor.empty() : tensor<1x16x112x112xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32>
    %6 = tensor.empty() : tensor<2x2xf32>
    %7 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%3, %6 : tensor<1x16x224x224xf32>, tensor<2x2xf32>) outs(%5 : tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32>
    %padded_24 = tensor.pad %7 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x16x112x112xf32> to tensor<1x16x114x114xf32>
    %8 = tensor.empty() : tensor<1x32x112x112xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_5 : tensor<32xf32>) outs(%8 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x112x112xf32>
    %10 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_24, %cst_4 : tensor<1x16x114x114xf32>, tensor<32x16x3x3xf32>) outs(%9 : tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    %11 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10 : tensor<1x32x112x112xf32>) outs(%8 : tensor<1x32x112x112xf32>) {
    ^bb0(%in: f32, %out: f32):
      %62 = arith.cmpf ugt, %in, %cst_0 : f32
      %63 = arith.select %62, %in, %cst_0 : f32
      linalg.yield %63 : f32
    } -> tensor<1x32x112x112xf32>
    %12 = tensor.empty() : tensor<1x32x56x56xf32>
    %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<1x32x56x56xf32>) -> tensor<1x32x56x56xf32>
    %14 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%11, %6 : tensor<1x32x112x112xf32>, tensor<2x2xf32>) outs(%13 : tensor<1x32x56x56xf32>) -> tensor<1x32x56x56xf32>
    %padded_25 = tensor.pad %14 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x32x56x56xf32> to tensor<1x32x58x58xf32>
    %15 = tensor.empty() : tensor<1x64x56x56xf32>
    %16 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_7 : tensor<64xf32>) outs(%15 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x64x56x56xf32>
    %17 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_25, %cst_6 : tensor<1x32x58x58xf32>, tensor<64x32x3x3xf32>) outs(%16 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %18 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17 : tensor<1x64x56x56xf32>) outs(%15 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %62 = arith.cmpf ugt, %in, %cst_0 : f32
      %63 = arith.select %62, %in, %cst_0 : f32
      linalg.yield %63 : f32
    } -> tensor<1x64x56x56xf32>
    %padded_26 = tensor.pad %18 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_9 : tensor<64xf32>) outs(%15 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x64x56x56xf32>
    %20 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_26, %cst_8 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%19 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %21 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20 : tensor<1x64x56x56xf32>) outs(%15 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %62 = arith.cmpf ugt, %in, %cst_0 : f32
      %63 = arith.select %62, %in, %cst_0 : f32
      linalg.yield %63 : f32
    } -> tensor<1x64x56x56xf32>
    %22 = tensor.empty() : tensor<1x64x28x28xf32>
    %23 = linalg.fill ins(%cst : f32) outs(%22 : tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>
    %24 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%21, %6 : tensor<1x64x56x56xf32>, tensor<2x2xf32>) outs(%23 : tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>
    %padded_27 = tensor.pad %24 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x64x28x28xf32> to tensor<1x64x30x30xf32>
    %25 = tensor.empty() : tensor<1x128x28x28xf32>
    %26 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_11 : tensor<128xf32>) outs(%25 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x28x28xf32>
    %27 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_27, %cst_10 : tensor<1x64x30x30xf32>, tensor<128x64x3x3xf32>) outs(%26 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %28 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%27 : tensor<1x128x28x28xf32>) outs(%25 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %62 = arith.cmpf ugt, %in, %cst_0 : f32
      %63 = arith.select %62, %in, %cst_0 : f32
      linalg.yield %63 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_28 = tensor.pad %28 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %29 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_13 : tensor<128xf32>) outs(%25 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x28x28xf32>
    %30 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_28, %cst_12 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%29 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %31 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%30 : tensor<1x128x28x28xf32>) outs(%25 : tensor<1x128x28x28xf32>) {
    ^bb0(%in: f32, %out: f32):
      %62 = arith.cmpf ugt, %in, %cst_0 : f32
      %63 = arith.select %62, %in, %cst_0 : f32
      linalg.yield %63 : f32
    } -> tensor<1x128x28x28xf32>
    %32 = tensor.empty() : tensor<1x128x14x14xf32>
    %33 = linalg.fill ins(%cst : f32) outs(%32 : tensor<1x128x14x14xf32>) -> tensor<1x128x14x14xf32>
    %34 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%31, %6 : tensor<1x128x28x28xf32>, tensor<2x2xf32>) outs(%33 : tensor<1x128x14x14xf32>) -> tensor<1x128x14x14xf32>
    %padded_29 = tensor.pad %34 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x14x14xf32> to tensor<1x128x16x16xf32>
    %35 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15 : tensor<128xf32>) outs(%32 : tensor<1x128x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x14x14xf32>
    %36 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_29, %cst_14 : tensor<1x128x16x16xf32>, tensor<128x128x3x3xf32>) outs(%35 : tensor<1x128x14x14xf32>) -> tensor<1x128x14x14xf32>
    %37 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%36 : tensor<1x128x14x14xf32>) outs(%32 : tensor<1x128x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %62 = arith.cmpf ugt, %in, %cst_0 : f32
      %63 = arith.select %62, %in, %cst_0 : f32
      linalg.yield %63 : f32
    } -> tensor<1x128x14x14xf32>
    %padded_30 = tensor.pad %37 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x128x14x14xf32> to tensor<1x128x16x16xf32>
    %38 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_17 : tensor<128xf32>) outs(%32 : tensor<1x128x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x14x14xf32>
    %39 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_30, %cst_16 : tensor<1x128x16x16xf32>, tensor<128x128x3x3xf32>) outs(%38 : tensor<1x128x14x14xf32>) -> tensor<1x128x14x14xf32>
    %40 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%39 : tensor<1x128x14x14xf32>) outs(%32 : tensor<1x128x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %62 = arith.cmpf ugt, %in, %cst_0 : f32
      %63 = arith.select %62, %in, %cst_0 : f32
      linalg.yield %63 : f32
    } -> tensor<1x128x14x14xf32>
    %41 = tensor.empty() : tensor<1x128x7x7xf32>
    %42 = linalg.fill ins(%cst : f32) outs(%41 : tensor<1x128x7x7xf32>) -> tensor<1x128x7x7xf32>
    %43 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%40, %6 : tensor<1x128x14x14xf32>, tensor<2x2xf32>) outs(%42 : tensor<1x128x7x7xf32>) -> tensor<1x128x7x7xf32>
    %44 = linalg.fill ins(%cst_0 : f32) outs(%41 : tensor<1x128x7x7xf32>) -> tensor<1x128x7x7xf32>
    %45 = tensor.empty() : tensor<1x1xf32>
    %46 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%43, %45 : tensor<1x128x7x7xf32>, tensor<1x1xf32>) outs(%44 : tensor<1x128x7x7xf32>) -> tensor<1x128x7x7xf32>
    %collapsed = tensor.collapse_shape %46 [[0], [1, 2, 3]] : tensor<1x128x7x7xf32> into tensor<1x6272xf32>
    %47 = tensor.empty() : tensor<6272x1000xf32>
    %48 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%cst_18 : tensor<1000x6272xf32>) outs(%47 : tensor<6272x1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<6272x1000xf32>
    %49 = tensor.empty() : tensor<1x1000xf32>
    %50 = linalg.fill ins(%cst_0 : f32) outs(%49 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %51 = linalg.matmul ins(%collapsed, %48 : tensor<1x6272xf32>, tensor<6272x1000xf32>) outs(%50 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %52 = linalg.generic {indexing_maps = [#map5, #map6, #map3], iterator_types = ["parallel", "parallel"]} ins(%51, %cst_19 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%49 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %in_31: f32, %out: f32):
      %62 = arith.addf %in, %in_31 : f32
      linalg.yield %62 : f32
    } -> tensor<1x1000xf32>
    %53 = linalg.generic {indexing_maps = [#map5, #map3], iterator_types = ["parallel", "parallel"]} ins(%52 : tensor<1x1000xf32>) outs(%49 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      %62 = arith.cmpf ugt, %in, %cst_0 : f32
      %63 = arith.select %62, %in, %cst_0 : f32
      linalg.yield %63 : f32
    } -> tensor<1x1000xf32>
    %54 = tensor.empty() : tensor<1000x1000xf32>
    %55 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%cst_20 : tensor<1000x1000xf32>) outs(%54 : tensor<1000x1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1000x1000xf32>
    %56 = linalg.matmul ins(%53, %55 : tensor<1x1000xf32>, tensor<1000x1000xf32>) outs(%50 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %57 = linalg.generic {indexing_maps = [#map5, #map6, #map3], iterator_types = ["parallel", "parallel"]} ins(%56, %cst_21 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%49 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %in_31: f32, %out: f32):
      %62 = arith.addf %in, %in_31 : f32
      linalg.yield %62 : f32
    } -> tensor<1x1000xf32>
    %58 = linalg.generic {indexing_maps = [#map5, #map3], iterator_types = ["parallel", "parallel"]} ins(%57 : tensor<1x1000xf32>) outs(%49 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      %62 = arith.cmpf ugt, %in, %cst_0 : f32
      %63 = arith.select %62, %in, %cst_0 : f32
      linalg.yield %63 : f32
    } -> tensor<1x1000xf32>
    %59 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%cst_22 : tensor<1000x1000xf32>) outs(%54 : tensor<1000x1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1000x1000xf32>
    %60 = linalg.matmul ins(%58, %59 : tensor<1x1000xf32>, tensor<1000x1000xf32>) outs(%50 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %61 = linalg.generic {indexing_maps = [#map5, #map6, #map3], iterator_types = ["parallel", "parallel"]} ins(%60, %cst_23 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%49 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %in_31: f32, %out: f32):
      %62 = arith.addf %in, %in_31 : f32
      linalg.yield %62 : f32
    } -> tensor<1x1000xf32>
    return %61 : tensor<1x1000xf32>
  }
  
    func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
    func.func private @printFlops(f64)
    func.func private @printMemrefF32(tensor<*xf32>)

    func.func @main() {
        %d1 = arith.constant 1: index
        %d0 = arith.constant 0 : index
        %n = arith.constant 2: index
    
      scf.for %i = %d0 to %n step %d1 {
        %0 = func.call @nanoTime() : () -> i64
        %1 = func.call @forward() : () -> tensor<1x1000xf32>
        %2 = func.call @nanoTime() : () -> i64

        %3 = arith.subi %2, %0 : i64
        %4 = arith.uitofp %3 : i64 to f64
        func.call @printFlops(%4) : (f64) -> ()
        }
    
        return
    }}

