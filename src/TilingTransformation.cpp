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

Tiling::Tiling(SmallVector<unsigned, 6>* tileSize, SmallVector<AffineForOp, 6>* band){
    this->tileSize = *tileSize;
    this->band = *band;
}

void getTileSizes(ArrayRef<AffineForOp> band, SmallVectorImpl<unsigned> *tileSizes);

void Tiling::applyTransformation(CodeIR CodeIr) {
    


    SmallVector<AffineForOp, 6> tiledNest;
    if (failed(tilePerfectlyNested(this->band, this->tileSize, &tiledNest))) {
            std::cout<<"FAILED\n";
    }

    mlir::OwningOpRef<Operation*>* module = (mlir::OwningOpRef<Operation*>*)CodeIr.getIr();

    mlir::PassManager pm((*module).get()->getName());

    // // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);
    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();    
    
    // // pm.addPass(mlir::createInlinerPass());
    // optPM.addPass(mlir::createConvertLinalgToAffineLoopsPass ()); 
    // //optPM.addPass(mlir::createLoopTilingPass ());
    // // for (Pass &pass : optPM.getPasses()) {
    // //     pass.initializeOptions("tile-size="+std::to_string(tileSize));
    // //     //pass.printAsTextualPipeline(output);
    // // }
    // // optPM.addPass(mlir::createConvertLinalgToLoopsPass());
    optPM.addPass(mlir::createConvertSCFToCFPass());
    //opPM.addPass(mlir::createConvertLinalgToLLVMPass ());
    optPM.addPass(memref::createExpandStridedMetadataPass ());
    optPM.addPass(mlir::createLowerAffinePass ());
    
    optPM.addPass( mlir::createArithToLLVMConversionPass());
    optPM.addPass(mlir::createConvertSCFToCFPass());
  
     
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass ());

    if (!mlir::failed(pm.run(*(*module))));
    
    // (**module)->dump();
    
    // CodeIr.setIr(module);
}

SmallVector<Node* , 2> Tiling::createTilingCandidates(MLIRCodeIR *CodeIr){
    
    std::vector<Transformation*>  TransformationList;
    // Bands of loops to tile.
    std::vector<SmallVector<AffineForOp, 6>> bands;
    
    std::vector<int> tileSizeoptions{ 4, 16, 32, 64 };

    Operation* op = ((mlir::OwningOpRef<Operation*>*)(*CodeIr).getIr())->get();
 
    SmallVector<AffineForOp , 2> forOps;
    SmallVector<MLIRCodeIR* , 2> CodeIRs;
    op->walk([&](AffineForOp forOp) { 
        MLIRCodeIR* ClonedCode =  (MLIRCodeIR*)CodeIr->cloneIr();
        CodeIRs.push_back(ClonedCode);
        forOps.push_back(forOp); 
        
        });
    SmallVector<AffineForOp , 2> forOpsToExplore;
    SmallVector<Node* , 2> ChildNodes;
    for (int i = 0; i < CodeIRs.size(); ++i) {
        auto& CloneCodeIr = CodeIRs[i];
        for (int size: tileSizeoptions){
            Node* ChildNode = new Node (CloneCodeIr);
            ChildNodes.push_back(ChildNode);
        }
        
        Operation* op = ((mlir::OwningOpRef<Operation*>*)(*CloneCodeIr).getIr())->get();

        SmallVector<AffineForOp , 2> forOps;
        op->walk([&](AffineForOp forOp) { forOps.push_back(forOp); });
        forOpsToExplore.push_back(forOps[i]);
    }

    for (auto forOp : forOpsToExplore) {
        SmallVector<AffineForOp, 6> nest;

        // Block &body = forOp.getRegion().front();
        // body.dump();
        // Get the maximal perfect nest.
        getPerfectlyNestedLoops(nest, forOp);        
        bands.push_back(nest);
    }

    for (int i = 0; i < bands.size(); ++i) {
        auto& band = bands[i];
        // if (!checkTilingLegality(band)) {
        // band.front().emitRemark("tiling code is illegal due to dependences");
        // continue;
        // }

        // Set up tile sizes; fill missing tile sizes at the end with default tile
        // size or tileSize if one was provided.
       
        
        for (int j = 0; j < tileSizeoptions.size(); ++j){
            SmallVector<unsigned, 6> tileSizes;
            getTileSizes(band, &tileSizes);
            std::optional<int64_t> fp = getMemoryFootprintBytes(band[0], 0);
            if (!fp) {
                std::cout<<"footprint \n";
                // Fill with default tile sizes if footprint is unknown.
                std::fill(tileSizes.begin(), tileSizes.end(),32);
            }
            Tiling *tiling = new Tiling(&tileSizes, &band);
            ChildNodes[i+j]->setTransformation(tiling);
        }
        
        //TransformationList.push_back(tiling);
        
        // for (const auto& size : tileSizes) {
        //     std::cout << size << " ";
        // }
        // std::cout << std::endl;
    } 
    return ChildNodes;
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