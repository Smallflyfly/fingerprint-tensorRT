#include <iostream>
#include "argsParser.h"
#include "common.h"

using namespace std;


class SampleOnnxFingerprint
{
	template <typename T>
	using SampleUniquePtr = unique_ptr<T, samplesCommon::InferDeleter>;
public:
	SampleOnnxFingerprint(const samplesCommon::OnnxSampleParams& params) : mParams(params), mEngine(nullptr)
	{

	}
private:
	samplesCommon::OnnxSampleParams mParams;
	shared_ptr<>
};