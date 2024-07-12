#map = affine_map<(d0, d1, d2, d3) -> (d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0) -> (d0 floordiv 49)>
#map4 = affine_map<(d0, d1) -> ((d0 floordiv 112) * 2 + (d1 mod 49) floordiv 7)>
#map5 = affine_map<(d0, d1) -> (d0 * 2 + d1 - (d0 floordiv 112) * 224 - (d1 floordiv 7) * 7)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

module attributes {torch.debug_module_name = "Net"} {
    memref.global "private" @global_seed : memref<i64> = dense<0>

    func.func @forward() -> tensor<32x64x12544xf32> {

      %0 = bufferization.alloc_tensor() : tensor<32x3x230x230xf32>
      %1 = bufferization.alloc_tensor() : tensor<64x3x7x7xf32>
      %2 = bufferization.alloc_tensor() : tensor<64xf32>
      %10 = bufferization.alloc_tensor() : tensor<10xf32>
      %11 = tensor.empty() : tensor<32x64x112x112xf32>
      
      %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<64xf32>) outs(%11 : tensor<32x64x112x112xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<32x64x112x112xf32>

      %collapsed = tensor.collapse_shape %1 [[0], [1, 2, 3]] : tensor<64x3x7x7xf32> into tensor<64x147xf32>
      %collapsed_1 = tensor.collapse_shape %12 [[0], [1], [2, 3]] : tensor<32x64x112x112xf32> into tensor<32x64x12544xf32>
      
      %13 = tensor.empty() : tensor<32x147x12544xf32>
      %14 = linalg.generic {producerTag, indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%13 : tensor<32x147x12544xf32>) {
      ^bb0(%out: f32):
        %50 = linalg.index 0 : index
        %51 = linalg.index 1 : index
        %52 = linalg.index 2 : index
        %53 = affine.apply #map3(%51)
        %54 = affine.apply #map4(%52, %51)
        %55 = affine.apply #map5(%52, %51)
        %extracted = tensor.extract %0[%50, %53, %54, %55] : tensor<32x3x230x230xf32>
        linalg.yield %extracted : f32
      } -> tensor<32x147x12544xf32>
      
      %15 = linalg.generic {consumerTag, indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed, %14 : tensor<64x147xf32>, tensor<32x147x12544xf32>) outs(%collapsed_1 : tensor<32x64x12544xf32>) {
      ^bb0(%in: f32, %in_6: f32, %out: f32):
        %50 = arith.mulf %in, %in_6 : f32
        %51 = arith.addf %50, %out : f32
        linalg.yield %51 : f32
      } -> tensor<32x64x12544xf32>
     
      return %15 : tensor<32x64x12544xf32>
    }

    func.func @main() {
      %1 = func.call @forward() : () -> tensor<32x64x12544xf32>
      return
    }
  }

