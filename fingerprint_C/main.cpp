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

#include <opencv2/opencv.hpp>

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

	bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
		SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
		SampleUniquePtr<nvonnxparser::IParser>& parser);

	bool processInput(const samplesCommon::BufferManager& buffers);

	bool verfiyOutput(const samplesCommon::BufferManager& buffers);



};

void showImage(string image)
{
	cv::Mat imageMat = cv::imread(image);
	cv::imshow("im", imageMat);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

bool SampleOnnxFingerprint::processInput(const samplesCommon::BufferManager& buffers)
{
	const int inputH = mInputDims.d[2];
	const int inputW = mInputDims.d[3];
	vector<uint8_t> fileData(inputH * inputW);

	//showImage("test.jpg");
	

	return true;
}

bool SampleOnnxFingerprint::infer()
{
	samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

	auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
	if (!context)
	{
		return false;
	}
	assert(mParams.inputTensorNames.size() == 1);
	cout << "hello1" << endl;
	if (!processInput(buffers))
	{
		return false;
	}
	return true;
}

bool SampleOnnxFingerprint::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
	SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config, SampleUniquePtr<nvonnxparser::IParser>& parser)
{
	auto parsed = parser->parseFromFile(
		locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(), static_cast<int>(gLogger.getReportableSeverity()));
	if (!parsed)
	{
		return false;
	}
	builder->setMaxBatchSize(mParams.batchSize);
	builder->setMaxWorkspaceSize(5_GiB);
	if (mParams.fp16)
	{
		config->setFlag(BuilderFlag::kFP16);
	}
	if (mParams.int8)
	{
		config->setFlag(BuilderFlag::kINT8);
		samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
	}
	samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

	return true;
}

bool SampleOnnxFingerprint::build()
{
	auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
	if (!builder)
	{
		return false;
	}
	const auto explictBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explictBatch));
	if (!network)
	{
		return false;
	}
	auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

	if (!config)
	{
		return false;
	}

	auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
	if (!parser)
	{
		return false;
	}

	auto constructed = constructNetwork(builder, network, config, parser);

	cout << "网络构建成功！" << endl;

	mEngine = shared_ptr<nvinfer1::ICudaEngine>(
		builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
	if (!mEngine)
	{
		return false;
	}

	assert(network->getNbInputs(0) == 1);
	mInputDims = network->getInput(0)->getDimensions();
	assert(mInputDims.nbDims == 4);

	assert(network->getNbOutputs == 1);
	mOutputDims = network->getOutput(0)->getDimensions();
	assert(mOutputDims.nbDims == 2);

	return true;
}


samplesCommon::OnnxSampleParams initFingerprintParams(const samplesCommon::Args& args)
{
	samplesCommon::OnnxSampleParams params;
	params.dataDirs.push_back("./");
	params.onnxFileName = "fingerprint.onnx";
	params.inputTensorNames.push_back("input");
	params.outputTensorNames.push_back("output");
	params.batchSize = 1;
	params.dlaCore = args.useDLACore;
	params.int8 = args.runInInt8;
	params.fp16 = args.runInFp16;
	//params.imageFilename = args.imageFilename;

	return params;
}

void printHelpInfo()
{
	std::cout
		<< "Usage: ./fingerprint_C/x64/fingerprint_C   [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
		<< std::endl;
	std::cout << "--help          Display help information" << std::endl;
	std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
		"where n is the number of DLA engines on the platform."
		<< std::endl;
	std::cout << "--int8          Run in Int8 mode." << std::endl;
	std::cout << "--fp16          Run in FP16 mode." << std::endl;
	std::cout << "--imageFile     image path like ./test.jpg" << std::endl;
}

int main(int argc, char** argv) {
	samplesCommon::Args args;
	cout << **argv << endl;
	bool argsOK = samplesCommon::parseArgs(args, argc, argv);
	if (!argsOK)
	{
		cout << "参数解析错误" << endl;
		return EXIT_FAILURE;
	}
	auto sampleTest = gLogger.defineTest("fingerprint", argc, argv);

	gLogger.reportTestStart(sampleTest);

	SampleOnnxFingerprint fingerprintSample(initFingerprintParams(args));

	gLogInfo << "Building and running a GPU inference engine for Onnx fingerprint" << endl;

	if (!fingerprintSample.build())
	{
		return gLogger.reportFail(sampleTest);
	}
	if (!fingerprintSample.infer())
	{
		return gLogger.reportFail(sampleTest);
	}

	return gLogger.reportPass(sampleTest);

	cout << "hello\n";
}