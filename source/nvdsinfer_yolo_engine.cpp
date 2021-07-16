/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "nvdsinfer_custom_impl.h"
#include "nvdsinfer_context.h"
#include "yolo.h"
#include "trt_utils.h"
#include <iostream>
#include <algorithm>

#define USE_CUDA_ENGINE_GET_API 1

static bool getYoloNetworkInfo (NetworkInfo &networkInfo, const NvDsInferContextInitParams* initParams)
{
    std::string yoloCfg = initParams->customNetworkConfigFilePath;
    std::string yoloType;

    std::transform (yoloCfg.begin(), yoloCfg.end(), yoloCfg.begin(), [] (uint8_t c) {
        return std::tolower (c);});

    if (yoloCfg.find("yolov5s") != std::string::npos) {
        if (yoloCfg == "yolov5s_p6") {
            yoloType = "yolov5s_p6";
        }
        else if (yoloCfg == "yolov5s_rep") {
            yoloType = "yolov5s_rep";
        }
        else {
            yoloType = "yolov5s";
        }
    } else if (yoloCfg.find("yolov5m") != std::string::npos) {
        if (yoloCfg == "yolov5m_p6") {
            yoloType = "yolov5m_p6";
        }
        else if (yoloCfg == "yolov5m_rep") {
            yoloType = "yolov5m_rep";
        }
        else {
            yoloType = "yolov5m";
        }
    } else if (yoloCfg.find("yolov5l") != std::string::npos) {
        if (yoloCfg == "yolov5l_p6") {
            yoloType = "yolov5l_p6";
        }
        else if (yoloCfg == "yolov5l_rep") {
            yoloType = "yolov5l_rep";
        }
        else {
            yoloType = "yolov5l";
        }
    } else if (yoloCfg.find("yolov5x") != std::string::npos) {
        if (yoloCfg == "yolov5x_p6") {
            yoloType = "yolov5x_p6";
        }
        else if (yoloCfg == "yolov5x_rep") {
            yoloType = "yolov5x_rep";
        }
        else {
            yoloType = "yolov5x";
        }
    } else {
        std::cerr << "Yolo type is not defined from config file name:"
                  << yoloCfg << std::endl;
        return false;
    }

    networkInfo.networkType     = yoloType;
    networkInfo.configFilePath  = initParams->customNetworkConfigFilePath;
    networkInfo.wtsFilePath     = initParams->modelFilePath;
    networkInfo.deviceType      = (initParams->useDLA ? "kDLA" : "kGPU");
    networkInfo.inputBlobName   = "data";

    if (networkInfo.configFilePath.empty() ||
        networkInfo.wtsFilePath.empty()) {
        std::cerr << "Yolo config file or weights file is NOT specified."
                  << std::endl;
        return false;
    }

    if (!fileExists(networkInfo.wtsFilePath)) {
        std::cerr << "Yolo config file or weights file is NOT exist."
                  << std::endl;
        return false;
    }

    return true;
}

#if !USE_CUDA_ENGINE_GET_API
IModelParser* NvDsInferCreateModelParser(
    const NvDsInferContextInitParams* initParams) {
    NetworkInfo networkInfo;
    if (!getYoloNetworkInfo(networkInfo, initParams)) {
      return nullptr;
    }

    return new Yolo(networkInfo);
}
#else
extern "C"
bool NvDsInferYoloCudaEngineGet(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine);

extern "C"
bool NvDsInferYoloCudaEngineGet(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine)
{
    NetworkInfo networkInfo;
    if (!getYoloNetworkInfo(networkInfo, initParams)) {
      return false;
    }

    Yolo yolo(networkInfo);
    cudaEngine = yolo.createEngine (builder);
    if (cudaEngine == nullptr)
    {
        std::cerr << "Failed to build cuda engine on "
                  << networkInfo.configFilePath << std::endl;
        return false;
    }

    return true;
}
#endif
