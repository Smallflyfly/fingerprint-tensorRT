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
#include <time.h>

#include <opencv2/opencv.hpp>

using namespace std;

const static int OUTPUT_SIZE = 512;


class SampleOnnxFingerprint
{
	template <typename T>
	using SampleUniquePtr = unique_ptr<T, samplesCommon::InferDeleter>;

public:
	SampleOnnxFingerprint(const samplesCommon::OnnxSampleParams& params) 
		: mParams(params)
	{
	}

	shared_ptr<nvinfer1::ICudaEngine> build();

	float* infer(samplesCommon::BufferManager& buffer, shared_ptr<nvinfer1::ICudaEngine> engine, string image);

private:
	samplesCommon::OnnxSampleParams mParams;

	

	nvinfer1::Dims mInputDims;
	nvinfer1::Dims mOutputDims;

	bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
		SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
		SampleUniquePtr<nvonnxparser::IParser>& parser);

	bool processInput(samplesCommon::BufferManager& buffers, string image);

	float* verfiyOutput(const samplesCommon::BufferManager& buffers);



};

void showImage(string image)
{
	cv::Mat imageMat = cv::imread(image);
	cv::imshow("im", imageMat);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

bool SampleOnnxFingerprint::processInput(samplesCommon::BufferManager& buffers, string image)
{
	const int inputC = mInputDims.d[1];
	const int inputH = mInputDims.d[2];
	const int inputW = mInputDims.d[3];
	const float imageMean = 0.568;
	const float imageStd = 0.389;
	uchar* fileDataChar = (uchar*)malloc(mParams.batchSize * inputC * inputH * inputW * sizeof(uchar));
	cv::Mat im = cv::imread(image, cv::COLOR_BGR2GRAY);
	cv::resize(im, im, cv::Size(inputH, inputW));
	//showImage("test.jpg");
	unsigned vol = inputH * inputW;
	fileDataChar = im.data;
	//float* fileData = (float*)malloc(mParams.batchSize * inputC * inputH * inputW * sizeof(float));
	float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
	// convert to 0-1
	for (int i = 0; i < vol; i++)
	{
		hostDataBuffer[i] = (float)fileDataChar[i] / 255.0;
	}
	// normalize
	// (0.568, 0.389)
	for (int i = 0; i < vol; i++)
	{
		hostDataBuffer[i] = (hostDataBuffer[i] - imageMean) / imageStd;
	}
	return true;
}

float* SampleOnnxFingerprint::verfiyOutput(const samplesCommon::BufferManager& buffers)
{
	const int outputSize = mOutputDims.d[1];
	float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
	/*for (int i = 0; i < 20; i++)
	{
		cout << output[i] << endl;
	}*/
	
	return output;
}

float* SampleOnnxFingerprint::infer(samplesCommon::BufferManager& buffers, shared_ptr<nvinfer1::ICudaEngine> engine, string image)
{
	auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
	assert(mParams.inputTensorNames.size() == 1);
	processInput(buffers, image);

	//Memcpy from host input buffers to device input buffers
	buffers.copyInputToDevice();

	bool status = context->executeV2(buffers.getDeviceBindings().data());
//	if (!status)
//	{
//		return false;
//	}

	// Memcpy from device output buffers to host output buffers
	buffers.copyOutputToHost();
	float* out = verfiyOutput(buffers);
	return out;
}

bool SampleOnnxFingerprint::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
	SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config, SampleUniquePtr<nvonnxparser::IParser>& parser)
{
	auto parsed = parser->parseFromFile(
		locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
	if (!parsed)
	{
		return false;
	}
	builder->setMaxBatchSize(mParams.batchSize);
	builder->setMaxWorkspaceSize(1_GiB);
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

shared_ptr<nvinfer1::ICudaEngine> SampleOnnxFingerprint::build()
{
	auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
//	if (!builder)
//	{
//		return false;
//	}
	const auto explictBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explictBatch));
//	if (!network)
//	{
//		return false;
//	}
	auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

//	if (!config)
//	{
//		return false;
//	}

	auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
//	if (!parser)
//	{
//		return false;
//	}

	auto constructed = constructNetwork(builder, network, config, parser);

//	cout << "���繹���ɹ���" << endl;

	shared_ptr<nvinfer1::ICudaEngine> mEngine = shared_ptr<nvinfer1::ICudaEngine>(
		builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
	

	assert(network->getNbInputs() == 1);
	mInputDims = network->getInput(0)->getDimensions();
	assert(mInputDims.nbDims == 4);

	assert(network->getNbOutputs() == 1);
	mOutputDims = network->getOutput(0)->getDimensions();
	assert(mOutputDims.nbDims == 2);

	cout << "build completed!" << endl;

	return mEngine;
}


samplesCommon::OnnxSampleParams initFingerprintParams(const samplesCommon::Args& args)
{
	samplesCommon::OnnxSampleParams params;
	params.dataDirs.push_back("data/");
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

float getMod(float* out)
{
	float sum = 0.0;
	for (int i = 0; i < OUTPUT_SIZE; i++)
	{
		sum += out[i] * out[i];
	}
	return sqrt(sum);
}

float getFingerprintSimlarity(float* out1, float* out2)
{
	float multi = 0.0;
	for (int i = 0; i < OUTPUT_SIZE; i++)
	{
		multi += out1[i] * out2[i];
	}
	return multi / (getMod(out1) * getMod(out2));
}

int main(int argc, char** argv) {
	samplesCommon::Args args;
	bool argsOK = samplesCommon::parseArgs(args, argc, argv);
	if (!argsOK)
	{
		cout << "������������" << endl;
		return EXIT_FAILURE;
	}
	auto sampleTest = sample::gLogger.defineTest("fingerprint", argc, argv);

	sample::gLogger.reportTestStart(sampleTest);

	SampleOnnxFingerprint fingerprintSample(initFingerprintParams(args));

	sample::gLogInfo << "Building and running a GPU inference engine for Onnx fingerprint" << endl;

	int batch_size = 1;
	string image_root = "data/";

	shared_ptr<nvinfer1::ICudaEngine> mEngine;
	mEngine = fingerprintSample.build();
	cout << "engine �����ɹ�" << endl;
	string image1 = image_root + "14.BMP";
	clock_t start, end;
	start = clock();
	samplesCommon::BufferManager buffers1(mEngine);
	float* out1 = fingerprintSample.infer(buffers1, mEngine, image1);

	string image2 = image_root + "2280.BMP";
	samplesCommon::BufferManager buffers2(mEngine);
	float* out2 = fingerprintSample.infer(buffers2, mEngine, image2);
	float simlarity = getFingerprintSimlarity(out1, out2);
	cout << "image1 image2 simlarity: " << simlarity << endl;
	end = clock();

	cout <<"cost time: " << end - start << endl;

	return sample::gLogger.reportPass(sampleTest);
}