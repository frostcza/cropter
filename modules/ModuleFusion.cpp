#include "ModuleFusion.h"

ModuleFusion::ModuleFusion(Precision p)
{
    string engine_name;
    if(p == INT8)
    {
        engine_name = "../../fusion/IFCNN_int8.engine";
    }
    else if(p == FP16)
    {
        engine_name = "../../fusion/IFCNN_fp16.engine";
    }
    ifcnn = new IFCNN(engine_name);
    printf("[Fusion] fusion engine initialize done \n");
}

ModuleFusion::~ModuleFusion()
{
    delete ifcnn;
}

void ModuleFusion::Fuse(uchar3* ir, uchar3* vi, void* fused_image)
{
    ifcnn->doInference(ir, vi, fused_image);
}