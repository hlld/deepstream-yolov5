/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>
#include <cassert>
#include <iostream>
#include "nvdsinfer_custom_impl.h"
#include "trt_utils.h"
#include "yololayer.h"

extern "C" bool NvDsInferParseCustomYoloV5(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseCustomYoloV5(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    UNUSED(networkInfo);
    UNUSED(detectionParams);

    const int det_size = sizeof(Yolo::Detection) / sizeof(float);
    const float* outputs = (const float *)(outputLayersInfo[0].buffer);
    int num = Yolo::MAX_OUTPUT_BBOX_COUNT;
    if (outputs[0] < Yolo::MAX_OUTPUT_BBOX_COUNT) {
        num = outputs[0];
    }
    if (num > 0) {
        objectList.resize(num);
        for (int k = 0; k < num; k++) {
            // Yolo::Detection format: [cx cy w h conf id]
            NvDsInferParseObjectInfo& info = objectList[k];
            const float* ptr = &outputs[1 + k * det_size];
            info.top = ptr[1] - ptr[3] / 2.F;
            info.left = ptr[0] - ptr[2] / 2.F;
            info.height = ptr[3];
            info.width = ptr[2];
            info.detectionConfidence = ptr[4];
            info.classId = ptr[5];
        }
    }
    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV5);
