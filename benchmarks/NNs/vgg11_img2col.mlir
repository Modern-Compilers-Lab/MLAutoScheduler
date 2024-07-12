#map = affine_map<(d0, d1, d2, d3) -> (d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0) -> (d0 floordiv 9)>
#map4 = affine_map<(d0) -> (d0 mod 9)>
#map5 = affine_map<(d0) -> ((d0 mod 9) floordiv 3)>
#map6 = affine_map<(d0) -> (d0 mod 3)>
#map7 = affine_map<(d0) -> (d0 floordiv 224)>
#map8 = affine_map<(d0) -> (d0 mod 224)>
#map9 = affine_map<(d0, d1) -> (d0 floordiv 224 + (d1 mod 9) floordiv 3)>
#map10 = affine_map<(d0, d1) -> (d0 + d1 - (d0 floordiv 224) * 224 - (d1 floordiv 3) * 3)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map14 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map15 = affine_map<(d0) -> (d0 floordiv 112)>
#map16 = affine_map<(d0) -> (d0 mod 112)>
#map17 = affine_map<(d0, d1) -> (d0 floordiv 112 + (d1 mod 9) floordiv 3)>
#map18 = affine_map<(d0, d1) -> (d0 + d1 - (d0 floordiv 112) * 112 - (d1 floordiv 3) * 3)>
#map19 = affine_map<(d0) -> (d0 floordiv 56)>
#map20 = affine_map<(d0) -> (d0 mod 56)>
#map21 = affine_map<(d0, d1) -> (d0 floordiv 56 + (d1 mod 9) floordiv 3)>
#map22 = affine_map<(d0, d1) -> (d0 + d1 - (d0 floordiv 56) * 56 - (d1 floordiv 3) * 3)>
#map23 = affine_map<(d0) -> (d0 floordiv 28)>
#map24 = affine_map<(d0) -> (d0 mod 28)>
#map25 = affine_map<(d0, d1) -> (d0 floordiv 28 + (d1 mod 9) floordiv 3)>
#map26 = affine_map<(d0, d1) -> (d0 + d1 - (d0 floordiv 28) * 28 - (d1 floordiv 3) * 3)>
#map27 = affine_map<(d0) -> (d0 floordiv 14)>
#map28 = affine_map<(d0) -> (d0 mod 14)>
#map29 = affine_map<(d0, d1) -> (d0 floordiv 14 + (d1 mod 9) floordiv 3)>
#map30 = affine_map<(d0, d1) -> (d0 + d1 - (d0 floordiv 14) * 14 - (d1 floordiv 3) * 3)>
#map31 = affine_map<(d0, d1) -> (d0, d1)>
#map32 = affine_map<(d0, d1) -> (d1, d0)>
#map33 = affine_map<(d0, d1) -> (0, d1)>
#map34 = affine_map<(d0, d1) -> (d1)>
  module attributes {torch.debug_module_name = "VGG11"} {
    memref.global "private" @global_seed : memref<i64> = dense<0>
    func.func @forward() -> tensor<1x1000xf32> {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0xFF800000 : f32
      %cst_0 = arith.constant 0.000000e+00 : f32
      %0 = bufferization.alloc_tensor() : tensor<1x3x224x224xf32>
      %1 = bufferization.alloc_tensor() : tensor<16x3x3x3xf32>
      %2 = bufferization.alloc_tensor() : tensor<16xf32>
      %3 = bufferization.alloc_tensor() : tensor<32x16x3x3xf32>
      %4 = bufferization.alloc_tensor() : tensor<32xf32>
      %5 = bufferization.alloc_tensor() : tensor<64x32x3x3xf32>
      %6 = bufferization.alloc_tensor() : tensor<64xf32>
      %7 = bufferization.alloc_tensor() : tensor<64x64x3x3xf32>
      %8 = bufferization.alloc_tensor() : tensor<64xf32>
      %9 = bufferization.alloc_tensor() : tensor<128x64x3x3xf32>
      %10 = bufferization.alloc_tensor() : tensor<128xf32>
      %11 = bufferization.alloc_tensor() : tensor<128x128x3x3xf32>
      %12 = bufferization.alloc_tensor() : tensor<128xf32>
      %13 = bufferization.alloc_tensor() : tensor<128x128x3x3xf32>
      %14 = bufferization.alloc_tensor() : tensor<128xf32>
      %15 = bufferization.alloc_tensor() : tensor<128x128x3x3xf32>
      %16 = bufferization.alloc_tensor() : tensor<128xf32>
      %17 = bufferization.alloc_tensor() : tensor<1000x6272xf32>
      %18 = bufferization.alloc_tensor() : tensor<1000xf32>
      %19 = bufferization.alloc_tensor() : tensor<1000x1000xf32>
      %20 = bufferization.alloc_tensor() : tensor<1000xf32>
      %21 = bufferization.alloc_tensor() : tensor<1000x1000xf32>
      %22 = bufferization.alloc_tensor() : tensor<1000xf32>
      %padded = tensor.pad %0 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
        tensor.yield %cst_0 : f32
      } : tensor<1x3x224x224xf32> to tensor<1x3x226x226xf32>
      %23 = tensor.empty() : tensor<1x16x224x224xf32>
      %24 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<16xf32>) outs(%23 : tensor<1x16x224x224xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x16x224x224xf32>
      %collapsed = tensor.collapse_shape %1 [[0], [1, 2, 3]] : tensor<16x3x3x3xf32> into tensor<16x27xf32>
      %collapsed_1 = tensor.collapse_shape %24 [[0], [1], [2, 3]] : tensor<1x16x224x224xf32> into tensor<1x16x50176xf32>
      %25 = tensor.empty() : tensor<1x27x50176xf32>
      %26 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%25 : tensor<1x27x50176xf32>) {
      ^bb0(%out: f32):
        %101 = linalg.index 0 : index
        %102 = linalg.index 1 : index
        %103 = linalg.index 2 : index
        %c3 = arith.constant 3 : index
        %c3_31 = arith.constant 3 : index
        %c3_32 = arith.constant 3 : index
        %c9 = arith.constant 9 : index
        %104 = affine.apply #map3(%102)
        %105 = affine.apply #map4(%102)
        %106 = affine.apply #map5(%102)
        %107 = affine.apply #map6(%102)
        %c224 = arith.constant 224 : index
        %c224_33 = arith.constant 224 : index
        %108 = affine.apply #map7(%103)
        %109 = affine.apply #map8(%103)
        %110 = affine.apply #map9(%103, %102)
        %111 = affine.apply #map10(%103, %102)
        %extracted = tensor.extract %padded[%101, %104, %110, %111] : tensor<1x3x226x226xf32>
        linalg.yield %extracted : f32
      } -> tensor<1x27x50176xf32>
      %27 = linalg.generic {indexing_maps = [#map11, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed, %26 : tensor<16x27xf32>, tensor<1x27x50176xf32>) outs(%collapsed_1 : tensor<1x16x50176xf32>) {
      ^bb0(%in: f32, %in_31: f32, %out: f32):
        %101 = arith.mulf %in, %in_31 : f32
        %102 = arith.addf %101, %out : f32
        linalg.yield %102 : f32
      } -> tensor<1x16x50176xf32>
      %expanded = tensor.expand_shape %27 [[0], [1], [2, 3]] : tensor<1x16x50176xf32> into tensor<1x16x224x224xf32>
      %28 = linalg.generic {indexing_maps = [#map14, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x16x224x224xf32>) outs(%23 : tensor<1x16x224x224xf32>) {
      ^bb0(%in: f32, %out: f32):
        %101 = arith.cmpf ugt, %in, %cst_0 : f32
        %102 = arith.select %101, %in, %cst_0 : f32
        linalg.yield %102 : f32
      } -> tensor<1x16x224x224xf32>
      %29 = tensor.empty() : tensor<1x16x112x112xf32>
      %30 = linalg.fill ins(%cst : f32) outs(%29 : tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32>
      %31 = tensor.empty() : tensor<2x2xf32>
      %32 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%28, %31 : tensor<1x16x224x224xf32>, tensor<2x2xf32>) outs(%30 : tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32>
      %padded_2 = tensor.pad %32 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
        tensor.yield %cst_0 : f32
      } : tensor<1x16x112x112xf32> to tensor<1x16x114x114xf32>
      %33 = tensor.empty() : tensor<1x32x112x112xf32>
      %34 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<32xf32>) outs(%33 : tensor<1x32x112x112xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x32x112x112xf32>
      %collapsed_3 = tensor.collapse_shape %3 [[0], [1, 2, 3]] : tensor<32x16x3x3xf32> into tensor<32x144xf32>
      %collapsed_4 = tensor.collapse_shape %34 [[0], [1], [2, 3]] : tensor<1x32x112x112xf32> into tensor<1x32x12544xf32>
      %35 = tensor.empty() : tensor<1x144x12544xf32>
      %36 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%35 : tensor<1x144x12544xf32>) {
      ^bb0(%out: f32):
        %101 = linalg.index 0 : index
        %102 = linalg.index 1 : index
        %103 = linalg.index 2 : index
        %c16 = arith.constant 16 : index
        %c3 = arith.constant 3 : index
        %c3_31 = arith.constant 3 : index
        %c9 = arith.constant 9 : index
        %104 = affine.apply #map3(%102)
        %105 = affine.apply #map4(%102)
        %106 = affine.apply #map5(%102)
        %107 = affine.apply #map6(%102)
        %c112 = arith.constant 112 : index
        %c112_32 = arith.constant 112 : index
        %108 = affine.apply #map15(%103)
        %109 = affine.apply #map16(%103)
        %110 = affine.apply #map17(%103, %102)
        %111 = affine.apply #map18(%103, %102)
        %extracted = tensor.extract %padded_2[%101, %104, %110, %111] : tensor<1x16x114x114xf32>
        linalg.yield %extracted : f32
      } -> tensor<1x144x12544xf32>
      %37 = linalg.generic {indexing_maps = [#map11, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_3, %36 : tensor<32x144xf32>, tensor<1x144x12544xf32>) outs(%collapsed_4 : tensor<1x32x12544xf32>) {
      ^bb0(%in: f32, %in_31: f32, %out: f32):
        %101 = arith.mulf %in, %in_31 : f32
        %102 = arith.addf %101, %out : f32
        linalg.yield %102 : f32
      } -> tensor<1x32x12544xf32>
      %expanded_5 = tensor.expand_shape %37 [[0], [1], [2, 3]] : tensor<1x32x12544xf32> into tensor<1x32x112x112xf32>
      %38 = linalg.generic {indexing_maps = [#map14, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_5 : tensor<1x32x112x112xf32>) outs(%33 : tensor<1x32x112x112xf32>) {
      ^bb0(%in: f32, %out: f32):
        %101 = arith.cmpf ugt, %in, %cst_0 : f32
        %102 = arith.select %101, %in, %cst_0 : f32
        linalg.yield %102 : f32
      } -> tensor<1x32x112x112xf32>
      %39 = tensor.empty() : tensor<1x32x56x56xf32>
      %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<1x32x56x56xf32>) -> tensor<1x32x56x56xf32>
      %41 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%38, %31 : tensor<1x32x112x112xf32>, tensor<2x2xf32>) outs(%40 : tensor<1x32x56x56xf32>) -> tensor<1x32x56x56xf32>
      %padded_6 = tensor.pad %41 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
        tensor.yield %cst_0 : f32
      } : tensor<1x32x56x56xf32> to tensor<1x32x58x58xf32>
      %42 = tensor.empty() : tensor<1x64x56x56xf32>
      %43 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<64xf32>) outs(%42 : tensor<1x64x56x56xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x64x56x56xf32>
      %collapsed_7 = tensor.collapse_shape %5 [[0], [1, 2, 3]] : tensor<64x32x3x3xf32> into tensor<64x288xf32>
      %collapsed_8 = tensor.collapse_shape %43 [[0], [1], [2, 3]] : tensor<1x64x56x56xf32> into tensor<1x64x3136xf32>
      %44 = tensor.empty() : tensor<1x288x3136xf32>
      %45 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%44 : tensor<1x288x3136xf32>) {
      ^bb0(%out: f32):
        %101 = linalg.index 0 : index
        %102 = linalg.index 1 : index
        %103 = linalg.index 2 : index
        %c32 = arith.constant 32 : index
        %c3 = arith.constant 3 : index
        %c3_31 = arith.constant 3 : index
        %c9 = arith.constant 9 : index
        %104 = affine.apply #map3(%102)
        %105 = affine.apply #map4(%102)
        %106 = affine.apply #map5(%102)
        %107 = affine.apply #map6(%102)
        %c56 = arith.constant 56 : index
        %c56_32 = arith.constant 56 : index
        %108 = affine.apply #map19(%103)
        %109 = affine.apply #map20(%103)
        %110 = affine.apply #map21(%103, %102)
        %111 = affine.apply #map22(%103, %102)
        %extracted = tensor.extract %padded_6[%101, %104, %110, %111] : tensor<1x32x58x58xf32>
        linalg.yield %extracted : f32
      } -> tensor<1x288x3136xf32>
      %46 = linalg.generic {indexing_maps = [#map11, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_7, %45 : tensor<64x288xf32>, tensor<1x288x3136xf32>) outs(%collapsed_8 : tensor<1x64x3136xf32>) {
      ^bb0(%in: f32, %in_31: f32, %out: f32):
        %101 = arith.mulf %in, %in_31 : f32
        %102 = arith.addf %101, %out : f32
        linalg.yield %102 : f32
      } -> tensor<1x64x3136xf32>
      %expanded_9 = tensor.expand_shape %46 [[0], [1], [2, 3]] : tensor<1x64x3136xf32> into tensor<1x64x56x56xf32>
      %47 = linalg.generic {indexing_maps = [#map14, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_9 : tensor<1x64x56x56xf32>) outs(%42 : tensor<1x64x56x56xf32>) {
      ^bb0(%in: f32, %out: f32):
        %101 = arith.cmpf ugt, %in, %cst_0 : f32
        %102 = arith.select %101, %in, %cst_0 : f32
        linalg.yield %102 : f32
      } -> tensor<1x64x56x56xf32>
      %padded_10 = tensor.pad %47 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
        tensor.yield %cst_0 : f32
      } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
      %48 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8 : tensor<64xf32>) outs(%42 : tensor<1x64x56x56xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x64x56x56xf32>
      %collapsed_11 = tensor.collapse_shape %7 [[0], [1, 2, 3]] : tensor<64x64x3x3xf32> into tensor<64x576xf32>
      %collapsed_12 = tensor.collapse_shape %48 [[0], [1], [2, 3]] : tensor<1x64x56x56xf32> into tensor<1x64x3136xf32>
      %49 = tensor.empty() : tensor<1x576x3136xf32>
      %50 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%49 : tensor<1x576x3136xf32>) {
      ^bb0(%out: f32):
        %101 = linalg.index 0 : index
        %102 = linalg.index 1 : index
        %103 = linalg.index 2 : index
        %c64 = arith.constant 64 : index
        %c3 = arith.constant 3 : index
        %c3_31 = arith.constant 3 : index
        %c9 = arith.constant 9 : index
        %104 = affine.apply #map3(%102)
        %105 = affine.apply #map4(%102)
        %106 = affine.apply #map5(%102)
        %107 = affine.apply #map6(%102)
        %c56 = arith.constant 56 : index
        %c56_32 = arith.constant 56 : index
        %108 = affine.apply #map19(%103)
        %109 = affine.apply #map20(%103)
        %110 = affine.apply #map21(%103, %102)
        %111 = affine.apply #map22(%103, %102)
        %extracted = tensor.extract %padded_10[%101, %104, %110, %111] : tensor<1x64x58x58xf32>
        linalg.yield %extracted : f32
      } -> tensor<1x576x3136xf32>
      %51 = linalg.generic {indexing_maps = [#map11, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_11, %50 : tensor<64x576xf32>, tensor<1x576x3136xf32>) outs(%collapsed_12 : tensor<1x64x3136xf32>) {
      ^bb0(%in: f32, %in_31: f32, %out: f32):
        %101 = arith.mulf %in, %in_31 : f32
        %102 = arith.addf %101, %out : f32
        linalg.yield %102 : f32
      } -> tensor<1x64x3136xf32>
      %expanded_13 = tensor.expand_shape %51 [[0], [1], [2, 3]] : tensor<1x64x3136xf32> into tensor<1x64x56x56xf32>
      %52 = linalg.generic {indexing_maps = [#map14, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_13 : tensor<1x64x56x56xf32>) outs(%42 : tensor<1x64x56x56xf32>) {
      ^bb0(%in: f32, %out: f32):
        %101 = arith.cmpf ugt, %in, %cst_0 : f32
        %102 = arith.select %101, %in, %cst_0 : f32
        linalg.yield %102 : f32
      } -> tensor<1x64x56x56xf32>
      %53 = tensor.empty() : tensor<1x64x28x28xf32>
      %54 = linalg.fill ins(%cst : f32) outs(%53 : tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>
      %55 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%52, %31 : tensor<1x64x56x56xf32>, tensor<2x2xf32>) outs(%54 : tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>
      %padded_14 = tensor.pad %55 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
        tensor.yield %cst_0 : f32
      } : tensor<1x64x28x28xf32> to tensor<1x64x30x30xf32>
      %56 = tensor.empty() : tensor<1x128x28x28xf32>
      %57 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10 : tensor<128xf32>) outs(%56 : tensor<1x128x28x28xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x128x28x28xf32>
      %collapsed_15 = tensor.collapse_shape %9 [[0], [1, 2, 3]] : tensor<128x64x3x3xf32> into tensor<128x576xf32>
      %collapsed_16 = tensor.collapse_shape %57 [[0], [1], [2, 3]] : tensor<1x128x28x28xf32> into tensor<1x128x784xf32>
      %58 = tensor.empty() : tensor<1x576x784xf32>
      %59 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%58 : tensor<1x576x784xf32>) {
      ^bb0(%out: f32):
        %101 = linalg.index 0 : index
        %102 = linalg.index 1 : index
        %103 = linalg.index 2 : index
        %c64 = arith.constant 64 : index
        %c3 = arith.constant 3 : index
        %c3_31 = arith.constant 3 : index
        %c9 = arith.constant 9 : index
        %104 = affine.apply #map3(%102)
        %105 = affine.apply #map4(%102)
        %106 = affine.apply #map5(%102)
        %107 = affine.apply #map6(%102)
        %c28 = arith.constant 28 : index
        %c28_32 = arith.constant 28 : index
        %108 = affine.apply #map23(%103)
        %109 = affine.apply #map24(%103)
        %110 = affine.apply #map25(%103, %102)
        %111 = affine.apply #map26(%103, %102)
        %extracted = tensor.extract %padded_14[%101, %104, %110, %111] : tensor<1x64x30x30xf32>
        linalg.yield %extracted : f32
      } -> tensor<1x576x784xf32>
      %60 = linalg.generic {indexing_maps = [#map11, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_15, %59 : tensor<128x576xf32>, tensor<1x576x784xf32>) outs(%collapsed_16 : tensor<1x128x784xf32>) {
      ^bb0(%in: f32, %in_31: f32, %out: f32):
        %101 = arith.mulf %in, %in_31 : f32
        %102 = arith.addf %101, %out : f32
        linalg.yield %102 : f32
      } -> tensor<1x128x784xf32>
      %expanded_17 = tensor.expand_shape %60 [[0], [1], [2, 3]] : tensor<1x128x784xf32> into tensor<1x128x28x28xf32>
      %61 = linalg.generic {indexing_maps = [#map14, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_17 : tensor<1x128x28x28xf32>) outs(%56 : tensor<1x128x28x28xf32>) {
      ^bb0(%in: f32, %out: f32):
        %101 = arith.cmpf ugt, %in, %cst_0 : f32
        %102 = arith.select %101, %in, %cst_0 : f32
        linalg.yield %102 : f32
      } -> tensor<1x128x28x28xf32>
      %padded_18 = tensor.pad %61 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
        tensor.yield %cst_0 : f32
      } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
      %62 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12 : tensor<128xf32>) outs(%56 : tensor<1x128x28x28xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x128x28x28xf32>
      %collapsed_19 = tensor.collapse_shape %11 [[0], [1, 2, 3]] : tensor<128x128x3x3xf32> into tensor<128x1152xf32>
      %collapsed_20 = tensor.collapse_shape %62 [[0], [1], [2, 3]] : tensor<1x128x28x28xf32> into tensor<1x128x784xf32>
      %63 = tensor.empty() : tensor<1x1152x784xf32>
      %64 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%63 : tensor<1x1152x784xf32>) {
      ^bb0(%out: f32):
        %101 = linalg.index 0 : index
        %102 = linalg.index 1 : index
        %103 = linalg.index 2 : index
        %c128 = arith.constant 128 : index
        %c3 = arith.constant 3 : index
        %c3_31 = arith.constant 3 : index
        %c9 = arith.constant 9 : index
        %104 = affine.apply #map3(%102)
        %105 = affine.apply #map4(%102)
        %106 = affine.apply #map5(%102)
        %107 = affine.apply #map6(%102)
        %c28 = arith.constant 28 : index
        %c28_32 = arith.constant 28 : index
        %108 = affine.apply #map23(%103)
        %109 = affine.apply #map24(%103)
        %110 = affine.apply #map25(%103, %102)
        %111 = affine.apply #map26(%103, %102)
        %extracted = tensor.extract %padded_18[%101, %104, %110, %111] : tensor<1x128x30x30xf32>
        linalg.yield %extracted : f32
      } -> tensor<1x1152x784xf32>
      %65 = linalg.generic {indexing_maps = [#map11, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_19, %64 : tensor<128x1152xf32>, tensor<1x1152x784xf32>) outs(%collapsed_20 : tensor<1x128x784xf32>) {
      ^bb0(%in: f32, %in_31: f32, %out: f32):
        %101 = arith.mulf %in, %in_31 : f32
        %102 = arith.addf %101, %out : f32
        linalg.yield %102 : f32
      } -> tensor<1x128x784xf32>
      %expanded_21 = tensor.expand_shape %65 [[0], [1], [2, 3]] : tensor<1x128x784xf32> into tensor<1x128x28x28xf32>
      %66 = linalg.generic {indexing_maps = [#map14, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_21 : tensor<1x128x28x28xf32>) outs(%56 : tensor<1x128x28x28xf32>) {
      ^bb0(%in: f32, %out: f32):
        %101 = arith.cmpf ugt, %in, %cst_0 : f32
        %102 = arith.select %101, %in, %cst_0 : f32
        linalg.yield %102 : f32
      } -> tensor<1x128x28x28xf32>
      %67 = tensor.empty() : tensor<1x128x14x14xf32>
      %68 = linalg.fill ins(%cst : f32) outs(%67 : tensor<1x128x14x14xf32>) -> tensor<1x128x14x14xf32>
      %69 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%66, %31 : tensor<1x128x28x28xf32>, tensor<2x2xf32>) outs(%68 : tensor<1x128x14x14xf32>) -> tensor<1x128x14x14xf32>
      %padded_22 = tensor.pad %69 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
        tensor.yield %cst_0 : f32
      } : tensor<1x128x14x14xf32> to tensor<1x128x16x16xf32>
      %70 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14 : tensor<128xf32>) outs(%67 : tensor<1x128x14x14xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x128x14x14xf32>
      %collapsed_23 = tensor.collapse_shape %13 [[0], [1, 2, 3]] : tensor<128x128x3x3xf32> into tensor<128x1152xf32>
      %collapsed_24 = tensor.collapse_shape %70 [[0], [1], [2, 3]] : tensor<1x128x14x14xf32> into tensor<1x128x196xf32>
      %71 = tensor.empty() : tensor<1x1152x196xf32>
      %72 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%71 : tensor<1x1152x196xf32>) {
      ^bb0(%out: f32):
        %101 = linalg.index 0 : index
        %102 = linalg.index 1 : index
        %103 = linalg.index 2 : index
        %c128 = arith.constant 128 : index
        %c3 = arith.constant 3 : index
        %c3_31 = arith.constant 3 : index
        %c9 = arith.constant 9 : index
        %104 = affine.apply #map3(%102)
        %105 = affine.apply #map4(%102)
        %106 = affine.apply #map5(%102)
        %107 = affine.apply #map6(%102)
        %c14 = arith.constant 14 : index
        %c14_32 = arith.constant 14 : index
        %108 = affine.apply #map27(%103)
        %109 = affine.apply #map28(%103)
        %110 = affine.apply #map29(%103, %102)
        %111 = affine.apply #map30(%103, %102)
        %extracted = tensor.extract %padded_22[%101, %104, %110, %111] : tensor<1x128x16x16xf32>
        linalg.yield %extracted : f32
      } -> tensor<1x1152x196xf32>
      %73 = linalg.generic {indexing_maps = [#map11, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_23, %72 : tensor<128x1152xf32>, tensor<1x1152x196xf32>) outs(%collapsed_24 : tensor<1x128x196xf32>) {
      ^bb0(%in: f32, %in_31: f32, %out: f32):
        %101 = arith.mulf %in, %in_31 : f32
        %102 = arith.addf %101, %out : f32
        linalg.yield %102 : f32
      } -> tensor<1x128x196xf32>
      %expanded_25 = tensor.expand_shape %73 [[0], [1], [2, 3]] : tensor<1x128x196xf32> into tensor<1x128x14x14xf32>
      %74 = linalg.generic {indexing_maps = [#map14, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_25 : tensor<1x128x14x14xf32>) outs(%67 : tensor<1x128x14x14xf32>) {
      ^bb0(%in: f32, %out: f32):
        %101 = arith.cmpf ugt, %in, %cst_0 : f32
        %102 = arith.select %101, %in, %cst_0 : f32
        linalg.yield %102 : f32
      } -> tensor<1x128x14x14xf32>
      %padded_26 = tensor.pad %74 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
        tensor.yield %cst_0 : f32
      } : tensor<1x128x14x14xf32> to tensor<1x128x16x16xf32>
      %75 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%16 : tensor<128xf32>) outs(%67 : tensor<1x128x14x14xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x128x14x14xf32>
      %collapsed_27 = tensor.collapse_shape %15 [[0], [1, 2, 3]] : tensor<128x128x3x3xf32> into tensor<128x1152xf32>
      %collapsed_28 = tensor.collapse_shape %75 [[0], [1], [2, 3]] : tensor<1x128x14x14xf32> into tensor<1x128x196xf32>
      %76 = tensor.empty() : tensor<1x1152x196xf32>
      %77 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%76 : tensor<1x1152x196xf32>) {
      ^bb0(%out: f32):
        %101 = linalg.index 0 : index
        %102 = linalg.index 1 : index
        %103 = linalg.index 2 : index
        %c128 = arith.constant 128 : index
        %c3 = arith.constant 3 : index
        %c3_31 = arith.constant 3 : index
        %c9 = arith.constant 9 : index
        %104 = affine.apply #map3(%102)
        %105 = affine.apply #map4(%102)
        %106 = affine.apply #map5(%102)
        %107 = affine.apply #map6(%102)
        %c14 = arith.constant 14 : index
        %c14_32 = arith.constant 14 : index
        %108 = affine.apply #map27(%103)
        %109 = affine.apply #map28(%103)
        %110 = affine.apply #map29(%103, %102)
        %111 = affine.apply #map30(%103, %102)
        %extracted = tensor.extract %padded_26[%101, %104, %110, %111] : tensor<1x128x16x16xf32>
        linalg.yield %extracted : f32
      } -> tensor<1x1152x196xf32>
      %78 = linalg.generic {indexing_maps = [#map11, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_27, %77 : tensor<128x1152xf32>, tensor<1x1152x196xf32>) outs(%collapsed_28 : tensor<1x128x196xf32>) {
      ^bb0(%in: f32, %in_31: f32, %out: f32):
        %101 = arith.mulf %in, %in_31 : f32
        %102 = arith.addf %101, %out : f32
        linalg.yield %102 : f32
      } -> tensor<1x128x196xf32>
      %expanded_29 = tensor.expand_shape %78 [[0], [1], [2, 3]] : tensor<1x128x196xf32> into tensor<1x128x14x14xf32>
      %79 = linalg.generic {indexing_maps = [#map14, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_29 : tensor<1x128x14x14xf32>) outs(%67 : tensor<1x128x14x14xf32>) {
      ^bb0(%in: f32, %out: f32):
        %101 = arith.cmpf ugt, %in, %cst_0 : f32
        %102 = arith.select %101, %in, %cst_0 : f32
        linalg.yield %102 : f32
      } -> tensor<1x128x14x14xf32>
      %80 = tensor.empty() : tensor<1x128x7x7xf32>
      %81 = linalg.fill ins(%cst : f32) outs(%80 : tensor<1x128x7x7xf32>) -> tensor<1x128x7x7xf32>
      %82 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%79, %31 : tensor<1x128x14x14xf32>, tensor<2x2xf32>) outs(%81 : tensor<1x128x7x7xf32>) -> tensor<1x128x7x7xf32>
      %83 = linalg.fill ins(%cst_0 : f32) outs(%80 : tensor<1x128x7x7xf32>) -> tensor<1x128x7x7xf32>
      %84 = tensor.empty() : tensor<1x1xf32>
      %85 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%82, %84 : tensor<1x128x7x7xf32>, tensor<1x1xf32>) outs(%83 : tensor<1x128x7x7xf32>) -> tensor<1x128x7x7xf32>
      %collapsed_30 = tensor.collapse_shape %85 [[0], [1, 2, 3]] : tensor<1x128x7x7xf32> into tensor<1x6272xf32>
      %86 = tensor.empty() : tensor<6272x1000xf32>
      %87 = linalg.generic {indexing_maps = [#map31, #map32], iterator_types = ["parallel", "parallel"]} ins(%17 : tensor<1000x6272xf32>) outs(%86 : tensor<6272x1000xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<6272x1000xf32>
      %88 = tensor.empty() : tensor<1x1000xf32>
      %89 = linalg.fill ins(%cst_0 : f32) outs(%88 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
      %90 = linalg.matmul ins(%collapsed_30, %87 : tensor<1x6272xf32>, tensor<6272x1000xf32>) outs(%89 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
      %91 = linalg.generic {indexing_maps = [#map33, #map34, #map31], iterator_types = ["parallel", "parallel"]} ins(%90, %18 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%88 : tensor<1x1000xf32>) {
      ^bb0(%in: f32, %in_31: f32, %out: f32):
        %101 = arith.addf %in, %in_31 : f32
        linalg.yield %101 : f32
      } -> tensor<1x1000xf32>
      %92 = linalg.generic {indexing_maps = [#map33, #map31], iterator_types = ["parallel", "parallel"]} ins(%91 : tensor<1x1000xf32>) outs(%88 : tensor<1x1000xf32>) {
      ^bb0(%in: f32, %out: f32):
        %101 = arith.cmpf ugt, %in, %cst_0 : f32
        %102 = arith.select %101, %in, %cst_0 : f32
        linalg.yield %102 : f32
      } -> tensor<1x1000xf32>
      %93 = tensor.empty() : tensor<1000x1000xf32>
      %94 = linalg.generic {indexing_maps = [#map31, #map32], iterator_types = ["parallel", "parallel"]} ins(%19 : tensor<1000x1000xf32>) outs(%93 : tensor<1000x1000xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1000x1000xf32>
      %95 = linalg.matmul ins(%92, %94 : tensor<1x1000xf32>, tensor<1000x1000xf32>) outs(%89 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
      %96 = linalg.generic {indexing_maps = [#map33, #map34, #map31], iterator_types = ["parallel", "parallel"]} ins(%95, %20 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%88 : tensor<1x1000xf32>) {
      ^bb0(%in: f32, %in_31: f32, %out: f32):
        %101 = arith.addf %in, %in_31 : f32
        linalg.yield %101 : f32
      } -> tensor<1x1000xf32>
      %97 = linalg.generic {indexing_maps = [#map33, #map31], iterator_types = ["parallel", "parallel"]} ins(%96 : tensor<1x1000xf32>) outs(%88 : tensor<1x1000xf32>) {
      ^bb0(%in: f32, %out: f32):
        %101 = arith.cmpf ugt, %in, %cst_0 : f32
        %102 = arith.select %101, %in, %cst_0 : f32
        linalg.yield %102 : f32
      } -> tensor<1x1000xf32>
      %98 = linalg.generic {indexing_maps = [#map31, #map32], iterator_types = ["parallel", "parallel"]} ins(%21 : tensor<1000x1000xf32>) outs(%93 : tensor<1000x1000xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1000x1000xf32>
      %99 = linalg.matmul ins(%97, %98 : tensor<1x1000xf32>, tensor<1000x1000xf32>) outs(%89 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
      %100 = linalg.generic {indexing_maps = [#map33, #map34, #map31], iterator_types = ["parallel", "parallel"]} ins(%99, %22 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%88 : tensor<1x1000xf32>) {
      ^bb0(%in: f32, %in_31: f32, %out: f32):
        %101 = arith.addf %in, %in_31 : f32
        linalg.yield %101 : f32
      } -> tensor<1x1000xf32>
      return %100 : tensor<1x1000xf32>
    }
    func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
    func.func private @printFlops(f64)
    func.func private @printMemrefF32(tensor<*xf32>)
    func.func @main() {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %0 = func.call @nanoTime() : () -> i64
        %1 = func.call @forward() : () -> tensor<1x1000xf32>
        %2 = func.call @nanoTime() : () -> i64
        %3 = arith.subi %2, %0 : i64
        %4 = arith.uitofp %3 : i64 to f64
        func.call @printFlops(%4) : (f64) -> ()
      }
      return
    }
  }

