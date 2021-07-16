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

#include "yolo.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <iterator>

extern void buildNetwork(INetworkDefinition* network, float gd, float gw, std::map<std::string, Weights>& weightMap, 
    std::string inputBlobName, std::string outputBlobName);
extern void buildNetwork_p6(INetworkDefinition* network, float gd, float gw, std::map<std::string, Weights>& weightMap, 
    std::string inputBlobName, std::string outputBlobName);
extern void buildNetwork_rep(INetworkDefinition* network, float gd, float gw, std::map<std::string, Weights>& weightMap, 
    std::string inputBlobName, std::string outputBlobName);
extern std::map<std::string, Weights> loadWeights(const std::string file);

Yolo::Yolo(const NetworkInfo& networkInfo)
    : m_NetworkType(networkInfo.networkType),
      m_ConfigFilePath(networkInfo.configFilePath),
      m_WtsFilePath(networkInfo.wtsFilePath),
      m_DeviceType(networkInfo.deviceType),
      m_InputBlobName(networkInfo.inputBlobName)
{
}

Yolo::~Yolo()
{
    destroyNetworkUtils();
}

nvinfer1::ICudaEngine *Yolo::createEngine(nvinfer1::IBuilder* builder)
{
    assert (builder);

    nvinfer1::INetworkDefinition *network = builder->createNetwork();
    if (parseModel(*network) != NVDSINFER_SUCCESS) {
        network->destroy();
        return nullptr;
    }

    // Build the engine
    std::cout << "Building the TensorRT Engine..." << std::endl;
    nvinfer1::ICudaEngine * engine = builder->buildCudaEngine(*network);
    if (engine) {
        std::cout << "Building complete!" << std::endl;
    } else {
        std::cerr << "Building engine failed!" << std::endl;
    }

    // destroy
    network->destroy();
    return engine;
}

NvDsInferStatus Yolo::parseModel(nvinfer1::INetworkDefinition& network) {
    destroyNetworkUtils();
    m_TrtWeights = loadWeights(m_WtsFilePath);

    // build yolo network
    std::cout << "Building Yolo network..." << std::endl;
    float gd = 1.0, gw = 1.0;
    if (m_NetworkType == "yolov5s") {
        gd = 0.33;
        gw = 0.50;
        buildNetwork(&network, gd, gw, m_TrtWeights, m_InputBlobName, "prob");
    }
    else if (m_NetworkType == "yolov5m") {
        gd = 0.67;
        gw = 0.75;
        buildNetwork(&network, gd, gw, m_TrtWeights, m_InputBlobName, "prob");
    }
    else if (m_NetworkType == "yolov5l") {
        gd = 1.0;
        gw = 1.0;
        buildNetwork(&network, gd, gw, m_TrtWeights, m_InputBlobName, "prob");
    }
    else if (m_NetworkType == "yolov5x") {
        gd = 1.33;
        gw = 1.25;
        buildNetwork(&network, gd, gw, m_TrtWeights, m_InputBlobName, "prob");
    }
    else if (m_NetworkType == "yolov5s_p6") {
        gd = 0.33;
        gw = 0.50;
        buildNetwork_p6(&network, gd, gw, m_TrtWeights, m_InputBlobName, "prob");
    }
    else if (m_NetworkType == "yolov5m_p6") {
        gd = 0.67;
        gw = 0.75;
        buildNetwork_p6(&network, gd, gw, m_TrtWeights, m_InputBlobName, "prob");
    }
    else if (m_NetworkType == "yolov5l_p6") {
        gd = 1.0;
        gw = 1.0;
        buildNetwork_p6(&network, gd, gw, m_TrtWeights, m_InputBlobName, "prob");
    }
    else if (m_NetworkType == "yolov5x_p6") {
        gd = 1.33;
        gw = 1.25;
        buildNetwork_p6(&network, gd, gw, m_TrtWeights, m_InputBlobName, "prob");
    }
    else if (m_NetworkType == "yolov5s_rep") {
        gd = 0.33;
        gw = 0.50;
        buildNetwork_rep(&network, gd, gw, m_TrtWeights, m_InputBlobName, "prob");
    }
    else if (m_NetworkType == "yolov5m_rep") {
        gd = 0.67;
        gw = 0.75;
        buildNetwork_rep(&network, gd, gw, m_TrtWeights, m_InputBlobName, "prob");
    }
    else if (m_NetworkType == "yolov5l_rep") {
        gd = 1.0;
        gw = 1.0;
        buildNetwork_rep(&network, gd, gw, m_TrtWeights, m_InputBlobName, "prob");
    }
    else if (m_NetworkType == "yolov5x_rep") {
        gd = 1.33;
        gw = 1.25;
        buildNetwork_rep(&network, gd, gw, m_TrtWeights, m_InputBlobName, "prob");
    }
    else {
        std::cout << "Building Yolo network failed!" << std::endl;
        return NVDSINFER_CONFIG_FAILED;
    }
    std::cout << "Building Yolo network complete!" << std::endl;

    return NVDSINFER_SUCCESS;
}

void Yolo::destroyNetworkUtils() {
    // deallocate the weights
    for (auto& mem : m_TrtWeights)
    {
        free(const_cast<void *>(mem.second.values));
    }
    m_TrtWeights.clear();
}