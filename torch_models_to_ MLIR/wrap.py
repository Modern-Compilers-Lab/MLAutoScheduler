import sys, re

file_name = sys.argv[1]
file_path = f'./mlir_files/{file_name}.mlir'

with open(file_path, 'r') as file:
    mlir_code = file.read()
    
end_module_idx = mlir_code.rfind('}')


pattern = r'tensor<([\dx]+)xf32>'

# Find all matches in the line
matches = re.findall(pattern, mlir_code)

if matches:
    input_shape = matches[0]  
    output_shape = matches[1]



new_text = f"""  
    func.func private @nanoTime() -> i64 attributes {{llvm.emit_c_interface}}
    func.func private @printFlops(f64)
    func.func private @printMemrefF32(tensor<*xf32>)

    func.func @main() {{
        %d1 = arith.constant 1: index
        %d0 = arith.constant 0 : index
        %n = arith.constant 2: index

          %val = arith.constant 2.00000e+00 : f32

          %out = bufferization.alloc_tensor() : tensor<{input_shape}xf32>
          %exin = linalg.fill ins(%val : f32) outs(%out : tensor<{input_shape}xf32>) -> tensor<{input_shape}xf32>
    
      //  scf.for %i = %d0 to %n step %d1 {{
        %0 = func.call @nanoTime() : () -> i64
        %1 = func.call @forward(%exin) : (tensor<{input_shape}xf32>) -> tensor<{output_shape}xf32>
        %2 = func.call @nanoTime() : () -> i64

        %unranked = tensor.cast %1 : tensor<{output_shape}xf32> to tensor<*xf32>
        func.call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()

        %3 = arith.subi %2, %0 : i64
        %4 = arith.uitofp %3 : i64 to f64
        func.call @printFlops(%4) : (f64) -> ()
      //  }}
    
        return
    }}"""

modified_content = mlir_code[:end_module_idx] + new_text + mlir_code[end_module_idx:]

with open(file_path, 'w') as file:
    file.write(modified_content)



