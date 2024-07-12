#map = affine_map<(d0, d1, d2, d3) -> (d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0) -> (d0 floordiv 49)>
#map4 = affine_map<(d0, d1) -> ((d0 floordiv 112) * 2 + (d1 mod 49) floordiv 7)>
#map5 = affine_map<(d0, d1) -> (d0 * 2 + d1 - (d0 floordiv 112) * 224 - (d1 floordiv 7) * 7)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map9 = affine_map<(d0) -> (d0 floordiv 25)>
#map10 = affine_map<(d0, d1) -> (d0 floordiv 52 + (d1 mod 25) floordiv 5)>
#map11 = affine_map<(d0, d1) -> (d0 + d1 - (d0 floordiv 52) * 52 - (d1 floordiv 5) * 5)>
#map12 = affine_map<(d0, d1) -> (d0, d1)>
#map13 = affine_map<(d0, d1) -> (d1, d0)>
#map14 = affine_map<(d0, d1) -> (d1)>

  module attributes {torch.debug_module_name = "Net"} {
    memref.global "private" @global_seed : memref<i64> = dense<0>
    func.func @forward() -> tensor<32x10xf32> {
      %cst = arith.constant 0xFF800000 : f32
      %cst_0 = arith.constant 0.000000e+00 : f32
      %0 = bufferization.alloc_tensor() : tensor<32x3x230x230xf32>
      %1 = bufferization.alloc_tensor() : tensor<64x3x7x7xf32>
      %2 = bufferization.alloc_tensor() : tensor<64xf32>
      %3 = bufferization.alloc_tensor() : tensor<16x64x5x5xf32>
      %4 = bufferization.alloc_tensor() : tensor<16xf32>
      %5 = bufferization.alloc_tensor() : tensor<120x10816xf32>
      %6 = bufferization.alloc_tensor() : tensor<120xf32>
      %7 = bufferization.alloc_tensor() : tensor<84x120xf32>
      %8 = bufferization.alloc_tensor() : tensor<84xf32>
      %9 = bufferization.alloc_tensor() : tensor<10x84xf32>
      %10 = bufferization.alloc_tensor() : tensor<10xf32>
      %11 = tensor.empty() : tensor<32x64x112x112xf32>
      
      %12 = linalg.generic {producerTag1,indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<64xf32>) outs(%11 : tensor<32x64x112x112xf32>) {
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
      %15 = linalg.generic {consumerTag,indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed, %14 : tensor<64x147xf32>, tensor<32x147x12544xf32>) outs(%collapsed_1 : tensor<32x64x12544xf32>) {
      ^bb0(%in: f32, %in_6: f32, %out: f32):
        %50 = arith.mulf %in, %in_6 : f32
        %51 = arith.addf %50, %out : f32
        linalg.yield %51 : f32
      } -> tensor<32x64x12544xf32>
      %expanded = tensor.expand_shape %15 [[0], [1], [2, 3]] : tensor<32x64x12544xf32> into tensor<32x64x112x112xf32>
      %16 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<32x64x112x112xf32>) outs(%11 : tensor<32x64x112x112xf32>) {
      ^bb0(%in: f32, %out: f32):
        %50 = arith.cmpf ugt, %in, %cst_0 : f32
        %51 = arith.select %50, %in, %cst_0 : f32
        linalg.yield %51 : f32
      } -> tensor<32x64x112x112xf32>
      %17 = tensor.empty() : tensor<32x64x56x56xf32>
      %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf32>
      %19 = tensor.empty() : tensor<2x2xf32>
      %20 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%16, %19 : tensor<32x64x112x112xf32>, tensor<2x2xf32>) outs(%18 : tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf32>
      %21 = tensor.empty() : tensor<32x16x52x52xf32>
      %22 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<16xf32>) outs(%21 : tensor<32x16x52x52xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<32x16x52x52xf32>
      %collapsed_2 = tensor.collapse_shape %3 [[0], [1, 2, 3]] : tensor<16x64x5x5xf32> into tensor<16x1600xf32>
      %collapsed_3 = tensor.collapse_shape %22 [[0], [1], [2, 3]] : tensor<32x16x52x52xf32> into tensor<32x16x2704xf32>
      %23 = tensor.empty() : tensor<32x1600x2704xf32>
      %24 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%23 : tensor<32x1600x2704xf32>) {
      ^bb0(%out: f32):
        %50 = linalg.index 0 : index
        %51 = linalg.index 1 : index
        %52 = linalg.index 2 : index
        %53 = affine.apply #map9(%51)
        %54 = affine.apply #map10(%52, %51)
        %55 = affine.apply #map11(%52, %51)
        %extracted = tensor.extract %20[%50, %53, %54, %55] : tensor<32x64x56x56xf32>
        linalg.yield %extracted : f32
      } -> tensor<32x1600x2704xf32>
      %25 = linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_2, %24 : tensor<16x1600xf32>, tensor<32x1600x2704xf32>) outs(%collapsed_3 : tensor<32x16x2704xf32>) {
      ^bb0(%in: f32, %in_6: f32, %out: f32):
        %50 = arith.mulf %in, %in_6 : f32
        %51 = arith.addf %50, %out : f32
        linalg.yield %51 : f32
      } -> tensor<32x16x2704xf32>
      %expanded_4 = tensor.expand_shape %25 [[0], [1], [2, 3]] : tensor<32x16x2704xf32> into tensor<32x16x52x52xf32>
      %26 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_4 : tensor<32x16x52x52xf32>) outs(%21 : tensor<32x16x52x52xf32>) {
      ^bb0(%in: f32, %out: f32):
        %50 = arith.cmpf ugt, %in, %cst_0 : f32
        %51 = arith.select %50, %in, %cst_0 : f32
        linalg.yield %51 : f32
      } -> tensor<32x16x52x52xf32>
      %27 = tensor.empty() : tensor<32x16x26x26xf32>
      %28 = linalg.fill ins(%cst : f32) outs(%27 : tensor<32x16x26x26xf32>) -> tensor<32x16x26x26xf32>
      %29 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%26, %19 : tensor<32x16x52x52xf32>, tensor<2x2xf32>) outs(%28 : tensor<32x16x26x26xf32>) -> tensor<32x16x26x26xf32>
      %collapsed_5 = tensor.collapse_shape %29 [[0], [1, 2, 3]] : tensor<32x16x26x26xf32> into tensor<32x10816xf32>
      %30 = tensor.empty() : tensor<10816x120xf32>
      %31 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<120x10816xf32>) outs(%30 : tensor<10816x120xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<10816x120xf32>
      %32 = tensor.empty() : tensor<32x120xf32>
      %33 = linalg.fill ins(%cst_0 : f32) outs(%32 : tensor<32x120xf32>) -> tensor<32x120xf32>
      %34 = linalg.matmul ins(%collapsed_5, %31 : tensor<32x10816xf32>, tensor<10816x120xf32>) outs(%33 : tensor<32x120xf32>) -> tensor<32x120xf32>
      %35 = linalg.generic {indexing_maps = [#map12, #map14, #map12], iterator_types = ["parallel", "parallel"]} ins(%34, %6 : tensor<32x120xf32>, tensor<120xf32>) outs(%32 : tensor<32x120xf32>) {
      ^bb0(%in: f32, %in_6: f32, %out: f32):
        %50 = arith.addf %in, %in_6 : f32
        linalg.yield %50 : f32
      } -> tensor<32x120xf32>
      %36 = linalg.generic {indexing_maps = [#map12, #map12], iterator_types = ["parallel", "parallel"]} ins(%35 : tensor<32x120xf32>) outs(%32 : tensor<32x120xf32>) {
      ^bb0(%in: f32, %out: f32):
        %50 = arith.cmpf ugt, %in, %cst_0 : f32
        %51 = arith.select %50, %in, %cst_0 : f32
        linalg.yield %51 : f32
      } -> tensor<32x120xf32>
      %37 = tensor.empty() : tensor<120x84xf32>
      %38 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<84x120xf32>) outs(%37 : tensor<120x84xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<120x84xf32>
      %39 = tensor.empty() : tensor<32x84xf32>
      %40 = linalg.fill ins(%cst_0 : f32) outs(%39 : tensor<32x84xf32>) -> tensor<32x84xf32>
      %41 = linalg.matmul ins(%36, %38 : tensor<32x120xf32>, tensor<120x84xf32>) outs(%40 : tensor<32x84xf32>) -> tensor<32x84xf32>
      %42 = linalg.generic {indexing_maps = [#map12, #map14, #map12], iterator_types = ["parallel", "parallel"]} ins(%41, %8 : tensor<32x84xf32>, tensor<84xf32>) outs(%39 : tensor<32x84xf32>) {
      ^bb0(%in: f32, %in_6: f32, %out: f32):
        %50 = arith.addf %in, %in_6 : f32
        linalg.yield %50 : f32
      } -> tensor<32x84xf32>
      %43 = linalg.generic {indexing_maps = [#map12, #map12], iterator_types = ["parallel", "parallel"]} ins(%42 : tensor<32x84xf32>) outs(%39 : tensor<32x84xf32>) {
      ^bb0(%in: f32, %out: f32):
        %50 = arith.cmpf ugt, %in, %cst_0 : f32
        %51 = arith.select %50, %in, %cst_0 : f32
        linalg.yield %51 : f32
      } -> tensor<32x84xf32>
      %44 = tensor.empty() : tensor<84x10xf32>
      %45 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<10x84xf32>) outs(%44 : tensor<84x10xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<84x10xf32>
      %46 = tensor.empty() : tensor<32x10xf32>
      %47 = linalg.fill ins(%cst_0 : f32) outs(%46 : tensor<32x10xf32>) -> tensor<32x10xf32>
      %48 = linalg.matmul ins(%43, %45 : tensor<32x84xf32>, tensor<84x10xf32>) outs(%47 : tensor<32x10xf32>) -> tensor<32x10xf32>
      %49 = linalg.generic {indexing_maps = [#map12, #map14, #map12], iterator_types = ["parallel", "parallel"]} ins(%48, %10 : tensor<32x10xf32>, tensor<10xf32>) outs(%46 : tensor<32x10xf32>) {
      ^bb0(%in: f32, %in_6: f32, %out: f32):
        %50 = arith.addf %in, %in_6 : f32
        linalg.yield %50 : f32
      } -> tensor<32x10xf32>
      return %49 : tensor<32x10xf32>
    }
    func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
    func.func private @printFlops(f64)
    func.func @main() {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %0 = func.call @nanoTime() : () -> i64
        %1 = func.call @forward() : () -> tensor<32x10xf32>
        %2 = func.call @nanoTime() : () -> i64
        %3 = arith.subi %2, %0 : i64
        %4 = arith.uitofp %3 : i64 to f64
        func.call @printFlops(%4) : (f64) -> ()
      }
      return
    }
  }





module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) 
{
  // %0 = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  //  %tiled_linalg_op= transform.structured.tile_using_for %0 [] interchange = [0, 2, 3, 1 , 5, 6 , 4]
  //   : (!transform.any_op) -> (!transform.any_op)

  // %conv = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  //   // expected-error@below {{failed to apply}}
  //   %res:2 = transform.structured.convert_conv2d_to_img2col %conv : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    
  //   //%1 = transform.structured.transpose_conv2d %0 : (!transform.any_op) -> (!transform.any_op)
  // %fb1 = transform.structured.match ops{["func.func"]} in %variant_op
  //     : (!transform.any_op) -> !transform.any_op
  //   transform.apply_patterns to %fb1 {
  //     transform.apply_patterns.canonicalization
  //   } : !transform.any_op
  //   transform.apply_cse to %fb1 : !transform.any_op


// %cons = transform.structured.match attributes{consumerTag} in %variant_op : (!transform.any_op) -> !transform.any_op
// transform.structured.interchange %cons iterator_interchange = [0, 2, 1, 3] : (!transform.any_op) -> !transform.any_op

// %consInter = transform.structured.match attributes{consumerTag} in %variant_op : (!transform.any_op) -> !transform.any_op
//    %conv_l1 , %forall_l1= transform.structured.tile_using_forall %consInter tile_sizes [ 4, 4,4, 7]
//   : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

// %prod = transform.structured.match attributes{producerTag} in %variant_op : (!transform.any_op) -> !transform.any_op 
//   transform.structured.interchange %prod iterator_interchange = [0, 2, 1] : (!transform.any_op) -> !transform.any_op
%prodInter = transform.structured.match attributes{producerTag} in %variant_op : (!transform.any_op) -> !transform.any_op 

   %conv_l1 , %forall_l1= transform.structured.tile_using_forall %prodInter tile_sizes [ 4, 7,4 ]
  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  // transform.structured.fuse_into_containing_op %prodInter into %forall_l1
  //   : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

//  %6 = transform.structured.match attributes{producerTag1} in %variant_op : (!transform.any_op) -> !transform.any_op 
//   transform.structured.fuse_into_containing_op %6 into %forall_l1
//   : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fb = transform.structured.match ops{["func.func"]} in %variant_op
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %fb {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %fb : !transform.any_op

    //  %n1 = transform.structured.match attributes{consumerTag} in %variant_op : (!transform.any_op) -> !transform.any_op
    //  transform.structured.vectorize %n1 : !transform.any_op 
      %n2 = transform.structured.match attributes{producerTag} in %variant_op : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %n2  : !transform.any_op 





//  %11 = transform.structured.match attributes{consumerTag1} in %variant_op : (!transform.any_op) -> !transform.any_op

// %conv_l11 , %forall_l11= transform.structured.tile_using_forall %11 tile_sizes [ 2, 2,2,2 ]
//     : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

//  %51 = transform.structured.match attributes{producerTag11} in %variant_op : (!transform.any_op) -> !transform.any_op  
  
//   transform.structured.fuse_into_containing_op %51 into %forall_l11
//    : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)



// %12 = transform.structured.match attributes{consumerTag2} in %variant_op : (!transform.any_op) -> !transform.any_op

// %conv_l12 , %forall_l12= transform.structured.tile_using_forall %12 tile_sizes [ 2, 2,16,16 ]
//     : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

//  %52 = transform.structured.match attributes{producerTag21} in %variant_op : (!transform.any_op) -> !transform.any_op  
  
//   transform.structured.fuse_into_containing_op %52 into %forall_l12
//    : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
//  %53 = transform.structured.match attributes{producerTag22} in %variant_op : (!transform.any_op) -> !transform.any_op  
  
//   transform.structured.fuse_into_containing_op %53 into %forall_l12
//    : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  

//     %fb = transform.structured.match ops{["func.func"]} in %variant_op
//       : (!transform.any_op) -> !transform.any_op
//     transform.apply_patterns to %fb {
//       transform.apply_patterns.canonicalization
//     } : !transform.any_op
//     transform.apply_cse to %fb : !transform.any_op

//   %n1 = transform.structured.match attributes{consumerTag} in %variant_op : (!transform.any_op) -> !transform.any_op

//   //%n2 = transform.structured.match attributes{producerTag} in %variant_op : (!transform.any_op) -> !transform.any_op
//   %n3 = transform.structured.match attributes{consumerTag1} in %variant_op : (!transform.any_op) -> !transform.any_op
//   //%n4 = transform.structured.match attributes{producerTag11} in %variant_op : (!transform.any_op) -> !transform.any_op
//   %n5 = transform.structured.match attributes{consumerTag2} in %variant_op : (!transform.any_op) -> !transform.any_op
//   %n6 = transform.structured.match attributes{producerTag21} in %variant_op : (!transform.any_op) -> !transform.any_op
//   %n7 = transform.structured.match attributes{producerTag22} in %variant_op : (!transform.any_op) -> !transform.any_op

//   transform.structured.vectorize %n1 : !transform.any_op 
//   //transform.structured.vectorize %n2 : !transform.any_op 
//   transform.structured.vectorize %n3 : !transform.any_op 
//   //transform.structured.vectorize %n4 : !transform.any_op 
//   transform.structured.vectorize %n5 : !transform.any_op 
//   transform.structured.vectorize %n6 : !transform.any_op 
//   transform.structured.vectorize %n7 : !transform.any_op 

//     %fb1 = transform.structured.match ops{["func.func"]} in %variant_op
//       : (!transform.any_op) -> !transform.any_op
//     transform.apply_patterns to %fb1 {
//       transform.apply_patterns.canonicalization
//     } : !transform.any_op
//     transform.apply_cse to %fb1 : !transform.any_op

// //Step 4. Vector backend
//   // ======================================================
//   %f = transform.structured.match ops{["func.func"]} in %variant_op
//     : (!transform.any_op) -> !transform.any_op

//   transform.apply_patterns to %f {


//     transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"

//     transform.apply_patterns.vector.transfer_permutation_patterns

//     transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerparallel"

//     transform.apply_patterns.vector.split_transfer_full_partial split_transfer_strategy = "vector-transfer"

//     transform.apply_patterns.vector.transfer_to_scf max_transfer_rank = 1 full_unroll = true

//     transform.apply_patterns.vector.lower_transfer max_transfer_rank = 1

//     transform.apply_patterns.vector.lower_shape_cast

//     transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_1d"

//     transform.apply_patterns.canonicalization
//   } : !transform.any_op
  transform.yield
}
}