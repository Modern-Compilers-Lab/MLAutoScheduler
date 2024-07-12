

    #map = affine_map<(d0, d1, d2, d3) -> (d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1, d0)>
#map5 = affine_map<(d0, d1) -> (0, d1)>
#map6 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "AlexNet"} {
  memref.global "private" @global_seed : memref<i64> = dense<0>
  func.func @forward() -> tensor<1x1000xf32> {
    %cst = arith.constant 4.000000e+00 : f32
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
     %cst_2 = bufferization.alloc_tensor() : tensor<1x3x256x256xf32>
    %cst_3 = bufferization.alloc_tensor() : tensor<32x3x11x11xf32>
    %cst_4 = bufferization.alloc_tensor() : tensor<32xf32>
    %cst_5 = bufferization.alloc_tensor() : tensor<64x32x5x5xf32>
    %cst_6 = bufferization.alloc_tensor() : tensor<64xf32>
    %cst_7 = bufferization.alloc_tensor() : tensor<192x64x3x3xf32>
    %cst_8 = bufferization.alloc_tensor() : tensor<192xf32>
    %cst_9 = bufferization.alloc_tensor() : tensor<64x192x3x3xf32>
    %cst_10 = bufferization.alloc_tensor() : tensor<64xf32>
    %cst_11 = bufferization.alloc_tensor() : tensor<64x64x3x3xf32>
    %cst_12 = bufferization.alloc_tensor() : tensor<64xf32>
    %cst_13 = bufferization.alloc_tensor() : tensor<1000x2304xf32>
    %cst_14 = bufferization.alloc_tensor() : tensor<1000xf32>
    %cst_15 = bufferization.alloc_tensor() : tensor<1000x1000xf32>
    %cst_16 = bufferization.alloc_tensor() : tensor<1000xf32>
    %cst_17 = bufferization.alloc_tensor() : tensor<1000x1000xf32>
    %cst_18 = bufferization.alloc_tensor() : tensor<1000xf32>
    %false = arith.constant false
    %padded = tensor.pad %cst_2 low[0, 0, 2, 2] high[0, 0, 2, 2] {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x3x256x256xf32> to tensor<1x3x260x260xf32>
    %0 = tensor.empty() : tensor<1x32x63x63xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_4 : tensor<32xf32>) outs(%0 : tensor<1x32x63x63xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x63x63xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<4> : vector<2xi64>} ins(%padded, %cst_3 : tensor<1x3x260x260xf32>, tensor<32x3x11x11xf32>) outs(%1 : tensor<1x32x63x63xf32>) -> tensor<1x32x63x63xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<1x32x63x63xf32>) outs(%0 : tensor<1x32x63x63xf32>) {
    ^bb0(%in: f32, %out: f32):
      %48 = arith.cmpf ugt, %in, %cst_1 : f32
      %49 = arith.select %48, %in, %cst_1 : f32
      linalg.yield %49 : f32
    } -> tensor<1x32x63x63xf32>
    %4 = tensor.empty() : tensor<1x32x31x31xf32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<1x32x31x31xf32>) -> tensor<1x32x31x31xf32>
    %6 = tensor.empty() : tensor<3x3xf32>
    %7 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%3, %6 : tensor<1x32x63x63xf32>, tensor<3x3xf32>) outs(%5 : tensor<1x32x31x31xf32>) -> tensor<1x32x31x31xf32>
    %padded_19 = tensor.pad %7 low[0, 0, 2, 2] high[0, 0, 2, 2] {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x32x31x31xf32> to tensor<1x32x35x35xf32>
    %8 = tensor.empty() : tensor<1x64x31x31xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_6 : tensor<64xf32>) outs(%8 : tensor<1x64x31x31xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x64x31x31xf32>
    %10 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_19, %cst_5 : tensor<1x32x35x35xf32>, tensor<64x32x5x5xf32>) outs(%9 : tensor<1x64x31x31xf32>) -> tensor<1x64x31x31xf32>
    %11 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10 : tensor<1x64x31x31xf32>) outs(%8 : tensor<1x64x31x31xf32>) {
    ^bb0(%in: f32, %out: f32):
      %48 = arith.cmpf ugt, %in, %cst_1 : f32
      %49 = arith.select %48, %in, %cst_1 : f32
      linalg.yield %49 : f32
    } -> tensor<1x64x31x31xf32>
    %12 = tensor.empty() : tensor<1x64x15x15xf32>
    %13 = linalg.fill ins(%cst_0 : f32) outs(%12 : tensor<1x64x15x15xf32>) -> tensor<1x64x15x15xf32>
    %14 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%11, %6 : tensor<1x64x31x31xf32>, tensor<3x3xf32>) outs(%13 : tensor<1x64x15x15xf32>) -> tensor<1x64x15x15xf32>
    %padded_20 = tensor.pad %14 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x64x15x15xf32> to tensor<1x64x17x17xf32>
    %15 = tensor.empty() : tensor<1x192x15x15xf32>
    %16 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_8 : tensor<192xf32>) outs(%15 : tensor<1x192x15x15xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x192x15x15xf32>
    %17 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_20, %cst_7 : tensor<1x64x17x17xf32>, tensor<192x64x3x3xf32>) outs(%16 : tensor<1x192x15x15xf32>) -> tensor<1x192x15x15xf32>
    %18 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17 : tensor<1x192x15x15xf32>) outs(%15 : tensor<1x192x15x15xf32>) {
    ^bb0(%in: f32, %out: f32):
      %48 = arith.cmpf ugt, %in, %cst_1 : f32
      %49 = arith.select %48, %in, %cst_1 : f32
      linalg.yield %49 : f32
    } -> tensor<1x192x15x15xf32>
    %padded_21 = tensor.pad %18 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x192x15x15xf32> to tensor<1x192x17x17xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_10 : tensor<64xf32>) outs(%12 : tensor<1x64x15x15xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x64x15x15xf32>
    %20 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_21, %cst_9 : tensor<1x192x17x17xf32>, tensor<64x192x3x3xf32>) outs(%19 : tensor<1x64x15x15xf32>) -> tensor<1x64x15x15xf32>
    %21 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20 : tensor<1x64x15x15xf32>) outs(%12 : tensor<1x64x15x15xf32>) {
    ^bb0(%in: f32, %out: f32):
      %48 = arith.cmpf ugt, %in, %cst_1 : f32
      %49 = arith.select %48, %in, %cst_1 : f32
      linalg.yield %49 : f32
    } -> tensor<1x64x15x15xf32>
    %padded_22 = tensor.pad %21 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x64x15x15xf32> to tensor<1x64x17x17xf32>
    %22 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_12 : tensor<64xf32>) outs(%12 : tensor<1x64x15x15xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x64x15x15xf32>
    %23 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded_22, %cst_11 : tensor<1x64x17x17xf32>, tensor<64x64x3x3xf32>) outs(%22 : tensor<1x64x15x15xf32>) -> tensor<1x64x15x15xf32>
    %24 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23 : tensor<1x64x15x15xf32>) outs(%12 : tensor<1x64x15x15xf32>) {
    ^bb0(%in: f32, %out: f32):
      %48 = arith.cmpf ugt, %in, %cst_1 : f32
      %49 = arith.select %48, %in, %cst_1 : f32
      linalg.yield %49 : f32
    } -> tensor<1x64x15x15xf32>
    %25 = tensor.empty() : tensor<1x64x7x7xf32>
    %26 = linalg.fill ins(%cst_0 : f32) outs(%25 : tensor<1x64x7x7xf32>) -> tensor<1x64x7x7xf32>
    %27 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%24, %6 : tensor<1x64x15x15xf32>, tensor<3x3xf32>) outs(%26 : tensor<1x64x7x7xf32>) -> tensor<1x64x7x7xf32>
    %28 = tensor.empty() : tensor<1x64x6x6xf32>
    %29 = linalg.fill ins(%cst_1 : f32) outs(%28 : tensor<1x64x6x6xf32>) -> tensor<1x64x6x6xf32>
    %30 = tensor.empty() : tensor<2x2xf32>
    %31 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%27, %30 : tensor<1x64x7x7xf32>, tensor<2x2xf32>) outs(%29 : tensor<1x64x6x6xf32>) -> tensor<1x64x6x6xf32>
    %32 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31 : tensor<1x64x6x6xf32>) outs(%28 : tensor<1x64x6x6xf32>) {
    ^bb0(%in: f32, %out: f32):
      %48 = arith.divf %in, %cst : f32
      linalg.yield %48 : f32
    } -> tensor<1x64x6x6xf32>
    %collapsed = tensor.collapse_shape %32 [[0], [1, 2, 3]] : tensor<1x64x6x6xf32> into tensor<1x2304xf32>
    %33 = tensor.empty() : tensor<2304x1000xf32>
    %34 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%cst_13 : tensor<1000x2304xf32>) outs(%33 : tensor<2304x1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<2304x1000xf32>
    %35 = tensor.empty() : tensor<1x1000xf32>
    %36 = linalg.fill ins(%cst_1 : f32) outs(%35 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %37 = linalg.matmul ins(%collapsed, %34 : tensor<1x2304xf32>, tensor<2304x1000xf32>) outs(%36 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %38 = linalg.generic {indexing_maps = [#map5, #map6, #map3], iterator_types = ["parallel", "parallel"]} ins(%37, %cst_14 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%35 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %in_23: f32, %out: f32):
      %48 = arith.addf %in, %in_23 : f32
      linalg.yield %48 : f32
    } -> tensor<1x1000xf32>
    %39 = linalg.generic {indexing_maps = [#map5, #map3], iterator_types = ["parallel", "parallel"]} ins(%38 : tensor<1x1000xf32>) outs(%35 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      %48 = arith.cmpf ugt, %in, %cst_1 : f32
      %49 = arith.select %48, %in, %cst_1 : f32
      linalg.yield %49 : f32
    } -> tensor<1x1000xf32>
    %40 = tensor.empty() : tensor<1000x1000xf32>
    %41 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%cst_15 : tensor<1000x1000xf32>) outs(%40 : tensor<1000x1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1000x1000xf32>
    %42 = linalg.matmul ins(%39, %41 : tensor<1x1000xf32>, tensor<1000x1000xf32>) outs(%36 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %43 = linalg.generic {indexing_maps = [#map5, #map6, #map3], iterator_types = ["parallel", "parallel"]} ins(%42, %cst_16 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%35 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %in_23: f32, %out: f32):
      %48 = arith.addf %in, %in_23 : f32
      linalg.yield %48 : f32
    } -> tensor<1x1000xf32>
    %44 = linalg.generic {indexing_maps = [#map5, #map3], iterator_types = ["parallel", "parallel"]} ins(%43 : tensor<1x1000xf32>) outs(%35 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      %48 = arith.cmpf ugt, %in, %cst_1 : f32
      %49 = arith.select %48, %in, %cst_1 : f32
      linalg.yield %49 : f32
    } -> tensor<1x1000xf32>
    %45 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%cst_17 : tensor<1000x1000xf32>) outs(%40 : tensor<1000x1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1000x1000xf32>
    %46 = linalg.matmul ins(%44, %45 : tensor<1x1000xf32>, tensor<1000x1000xf32>) outs(%36 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %47 = linalg.generic {indexing_maps = [#map5, #map6, #map3], iterator_types = ["parallel", "parallel"]} ins(%46, %cst_18 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%35 : tensor<1x1000xf32>) {
    ^bb0(%in: f32, %in_23: f32, %out: f32):
      %48 = arith.addf %in, %in_23 : f32
      linalg.yield %48 : f32
    } -> tensor<1x1000xf32>
    return %47 : tensor<1x1000xf32>
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

