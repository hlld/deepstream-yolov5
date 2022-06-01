#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <iostream>
#include <vector>
#include <string>
#include <vector>
#include <algorithm>
#include <cudnn.h>
#include "NvInfer.h"

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }
#endif

namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f;
    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT * 2];
    };
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 80;
    static constexpr int INPUT_H = 640;
    static constexpr int INPUT_W = 640;

    static constexpr int LOCATIONS = 4;
    struct alignas(float) Detection {
        float bbox[LOCATIONS];
        float conf;
        float class_id;
    };
}

namespace nvinfer1
{
    class YoloLayerPlugin : public IPluginV2
    {
    public:
        YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, 
            const std::vector<Yolo::YoloKernel>& vYoloKernel);
        YoloLayerPlugin(const void* data, size_t length);
        ~YoloLayerPlugin();

        const char* getPluginType () const noexcept override { return "YoloLayer_TRT"; }
        const char* getPluginVersion () const noexcept override { return "1"; }
        int getNbOutputs () const noexcept override { return 1; }

        nvinfer1::Dims getOutputDimensions (
            int index, const Dims* inputs,
            int nbInputDims) noexcept override;

        bool supportsFormat (
            DataType type, PluginFormat format) const noexcept override;

        void configureWithFormat (
            const Dims* inputDims, int nbInputs,
            const Dims* outputDims, int nbOutputs,
            DataType type, PluginFormat format, int maxBatchSize) noexcept override;

        int initialize () noexcept override { return 0; }
        void terminate () noexcept override {}
        size_t getWorkspaceSize (int maxBatchSize) const noexcept override { return 0; }
        int32_t enqueue (
            int32_t batchSize, void const* const* inputs, void* const* outputs,
            void* workspace, cudaStream_t stream) noexcept override;
        size_t getSerializationSize() const noexcept override;
        void serialize (void* buffer) const noexcept override;
        void destroy () noexcept override { delete this; }
        IPluginV2* clone() const noexcept override;

        void setPluginNamespace (const char* pluginNamespace) noexcept override {
            mNamespace = pluginNamespace;
        }
        virtual const char* getPluginNamespace () const noexcept override {
            return mNamespace.c_str();
        }

    private:
        std::string mNamespace;
        int mThreadCount = 256;
        int mKernelCount;
        int mClassCount;
        int mYoloV5NetWidth;
        int mYoloV5NetHeight;
        int mMaxOutObject;
        void **mAnchor;
        std::vector<Yolo::YoloKernel> mYoloKernel;
    };

    class YoloPluginCreator : public IPluginCreator
    {
    public:
        YoloPluginCreator();
        ~YoloPluginCreator() override = default;

        const char* getPluginName () const noexcept override { return "YoloLayer_TRT"; }
        const char* getPluginVersion () const noexcept override { return "1"; }

        const PluginFieldCollection* getFieldNames() noexcept override { return &mFC; };

        IPluginV2* createPlugin (
            const char* name, const PluginFieldCollection* fc) noexcept override;

        IPluginV2* deserializePlugin (
            const char* name, const void* serialData, size_t serialLength) noexcept override;

        void setPluginNamespace(const char* libNamespace) noexcept override {
            mNamespace = libNamespace;
        }
        const char* getPluginNamespace() const noexcept override {
            return mNamespace.c_str();
        }

    private:
        std::string mNamespace;
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
    };
    REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
};

#endif 
