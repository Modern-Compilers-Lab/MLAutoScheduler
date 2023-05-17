//===------------ TilingTransformation.cpp TilingTransformation -----------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implmentation of the TilingTransformation class, which  
/// contains the declartion of the tiling transformation
///
//===----------------------------------------------------------------------===//
#include "TilingTransformation.h"

using namespace mlir;

Tiling::Tiling(int tileSize){
    this->tileSize = tileSize;
}

void getTileSizes(ArrayRef<AffineForOp> band, SmallVectorImpl<unsigned> *tileSizes);

void Tiling::applyTransformation(CodeIR CodeIr) {
    
    mlir::OwningOpRef<Operation*>* module = (mlir::OwningOpRef<Operation*>*)CodeIr.getIr();

    mlir::PassManager pm((*module).get()->getName());

    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);
    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();    
    
    // pm.addPass(mlir::createInlinerPass());
    optPM.addPass(mlir::createConvertLinalgToAffineLoopsPass ()); 
    //optPM.addPass(mlir::createLoopTilingPass ());
    // for (Pass &pass : optPM.getPasses()) {
    //     pass.initializeOptions("tile-size="+std::to_string(tileSize));
    //     //pass.printAsTextualPipeline(output);
    // }
    // optPM.addPass(mlir::createConvertLinalgToLoopsPass());
    // optPM.addPass(mlir::createConvertSCFToCFPass());
    // //opPM.addPass(mlir::createConvertLinalgToLLVMPass ());
    // optPM.addPass(memref::createExpandStridedMetadataPass ());
    // optPM.addPass(mlir::createLowerAffinePass ());
    
    // optPM.addPass( mlir::createArithToLLVMConversionPass());
    // optPM.addPass(mlir::createConvertSCFToCFPass());
  
     
    // pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    // pm.addPass(mlir::createConvertFuncToLLVMPass());
    // pm.addPass(mlir::createReconcileUnrealizedCastsPass ());

    if (!mlir::failed(pm.run(*(*module))))
    
    (**module)->dump();
    
    CodeIr.setIr(module);
}

std::list<Transformation*> Tiling::createCandidates(CodeIR *CodeIr){
 
    // Bands of loops to tile.
    std::vector<SmallVector<AffineForOp, 6>> bands;
    
    Operation* op = ((mlir::OwningOpRef<Operation*>*)(*CodeIr).getIr())->get();
 
    SmallVector<AffineForOp, 2> forOps;
    op->walk([&](AffineForOp forOp) { forOps.push_back(forOp); });

    for (auto forOp : forOps) {
        SmallVector<AffineForOp, 6> nest;

        Block &body = forOp.getRegion().front();
        body.dump();
        // Get the maximal perfect nest.
        getPerfectlyNestedLoops(nest, forOp);        
        bands.push_back(nest);
    }

    for (auto &band : bands) {
        // if (!checkTilingLegality(band)) {
        // band.front().emitRemark("tiling code is illegal due to dependences");
        // continue;
        // }

        // Set up tile sizes; fill missing tile sizes at the end with default tile
        // size or tileSize if one was provided.
        SmallVector<unsigned, 6> tileSizes;

        getTileSizes(band, &tileSizes);
        std::optional<int64_t> fp = getMemoryFootprintBytes(band[0], 0);
        if (!fp) {
            std::cout<<"footprint \n";
            // Fill with default tile sizes if footprint is unknown.
            std::fill(tileSizes.begin(), tileSizes.end(),32);
        }
        for (const auto& size : tileSizes) {
            std::cout << size << " ";
        }
        std::cout << std::endl;
    }
    
    std::list<Transformation*>  list;
    return list;
}

void getTileSizes(ArrayRef<AffineForOp> band,
                              SmallVectorImpl<unsigned> *tileSizes) {
  if (band.empty())
    return;

  // Use tileSizes and fill them with default tile size if it's short.
  if (!tileSizes->empty()) {
    std::cout<<"inside \n";
    tileSizes->assign(tileSizes->begin(), tileSizes->end());
    tileSizes->resize(band.size(), 4);
    return;
  }
  tileSizes->resize(band.size());
}