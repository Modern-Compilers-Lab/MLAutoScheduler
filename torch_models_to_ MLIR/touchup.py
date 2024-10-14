import re
import sys

file_name = sys.argv[1]
file_path = f'./mlir_files/{file_name}.mlir'

with open(file_path, 'r') as file:
    mlir_code = file.readlines()

string_to_remove = "cf.assert %3, \"training is not supported for now\""

mlir_code = [line for line in mlir_code if string_to_remove not in line]
mlir_code = ''.join(mlir_code)

pattern = re.compile(r'ml_program\.global\s+private\s+mutable\s+@global_seed\s*\(\s*dense<0>\s*:\s*tensor<i64>\s*\)\s*:\s*tensor<i64>')
match = pattern.search(mlir_code[:mlir_code.find('\n', 0, 20)])

if match:
    mlir_code = mlir_code[:match.start()] + 'memref.global "private" @global_seed : memref<i64> = dense<0>' + mlir_code[match.end():]

with open(file_path, 'w') as file:
    file.write(mlir_code)

