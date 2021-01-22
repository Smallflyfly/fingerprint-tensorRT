#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
//
#include "NvInfer.h"
#include <cuda_runtime_api.h>
//
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;


class SampleOnnxFingerprint
{
	template <typename T>
	using SampleUniquePtr = unique_ptr<T, samplesCommon::InferDeleter>;

public:
	SampleOnnxFingerprint(const samplesCommon::OnnxSampleParams& params) 
		: mParams(params), mEngine(nullptr)
	{
	}

	bool build();

	bool infer();

private:
	samplesCommon::OnnxSampleParams mParams;
	shared_ptr<nvinfer1::ICudaEngine> mEngine;
	nvinfer1::Dims mInputDims;
	nvinfer1::Dims mOutputDims;

};

int main() {
	cout << "hello" << endl;
}