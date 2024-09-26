cmake_minimum_required(VERSION 3.13.4)
project(AutoSchedulerML LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")






if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
else()
  # Build via external projects mechanism
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
  set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
  set(MLIR_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)

  include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
  include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})
endif()

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

set(STANDALONE_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(STANDALONE_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${STANDALONE_SOURCE_DIR}/include)
include_directories(${STANDALONE_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})


get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)


# Add the include directories
include_directories(include)
include_directories(include/CustomPasses)
include_directories(coreAutoScheduler/include)

add_subdirectory(include)
# # Add the source files
file(GLOB SOURCES "src/*.cpp")
file(GLOB PASSES_SOURCES "src/CustomPasses/*.cpp")

# # Add the executable
# add_executable(AutoML ${SOURCES})

add_llvm_executable(AutoSchedulerML 
main.cpp
${PASSES_SOURCES}
${SOURCES}
DEPENDS
CustomPassesIncGen
)

llvm_update_compile_flags(AutoSchedulerML)

# Link the library
add_subdirectory(./coreAutoScheduler build)
target_link_libraries(AutoSchedulerML PUBLIC coreAutoScheduler)
target_link_libraries(AutoSchedulerML
PRIVATE
  ${dialect_libs}

  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTargetLLVMIRExport
  
  MLIRAffineDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIROptLib
  MLIRSCFToControlFlow

  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIROpenMPToLLVM
  MLIRSCFToOpenMP
  MLIRBufferizationToMemRef

  MLIRToLLVMIRTranslationRegistration
  MLIRLLVMToLLVMIRTranslation
  MLIRExecutionEngine
  MLIRJitRunner

  MLIRAnalysis
  MLIRSupport
  
  MLIRFunctionInterfaces
  MLIRTransformDialect
  MLIRTransforms
  MLIRTransformDialectUtils
  MLIRTransformDialectTransforms




  )
  #add_subdirectory(src)
mlir_check_all_link_libraries(AutoSchedulerML)
