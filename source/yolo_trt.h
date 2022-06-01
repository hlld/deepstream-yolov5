/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef _YOLO_TRT_H_
#define _YOLO_TRT_H_

#include <stdint.h>
#include <string>
#include <vector>
#include <map>
#include <memory>

#include "NvInfer.h"
#include "nvdsinfer_custom_impl.h"
using namespace nvinfer1;

/**
 * Holds all the file paths required to build a network.
 */
struct NetworkInfo
{
    std::string networkType;
    std::string configFilePath;
    std::string wtsFilePath;
    std::string deviceType;
    std::string inputBlobName;
};

class Yolo : public IModelParser {
public:
    Yolo(const NetworkInfo& networkInfo);
    ~Yolo() override;
    bool hasFullDimsSupported() const override { return false; }
    const char* getModelName() const override {
        return m_ConfigFilePath.empty() ? m_NetworkType.c_str()
                                        : m_ConfigFilePath.c_str();
    }
    NvDsInferStatus parseModel(nvinfer1::INetworkDefinition& network) override;

    nvinfer1::ICudaEngine *createEngine (
        nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config);

protected:
    const std::string m_NetworkType;
    const std::string m_ConfigFilePath;
    const std::string m_WtsFilePath;
    const std::string m_DeviceType;
    const std::string m_InputBlobName;

    // TRT specific members
    std::map<std::string, Weights> m_TrtWeights;

private:
    void destroyNetworkUtils();
};

void buildNetwork(INetworkDefinition* network, float gd, float gw, std::map<std::string, 
    Weights>& weightMap, std::string inputBlobName, std::string outputBlobName);
void buildNetwork_p6(INetworkDefinition* network, float gd, float gw, std::map<std::string, 
    Weights>& weightMap, std::string inputBlobName, std::string outputBlobName);
std::map<std::string, Weights> loadWeights(const std::string file);

#endif // _YOLO_TRT_H_
