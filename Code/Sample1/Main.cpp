#include <Windows.h>
#include "Layers/LinearLayer.h"
#include "Common/Meerkat.h"

using namespace DeepLearning;

int main()
{
	DeepLearning::Init();

	Tensor* input = DL_NEW(Tensor)(ComputeType_CPU, { 1, 2 });
	Tensor* output = DL_NEW(Tensor)(ComputeType_CPU, { 1, 1 });
	input->Zeros();

	LinearLayer* linear_layer = DL_NEW(LinearLayer)(ComputeType_CPU, 2, 1);
	linear_layer->Forward(input, output);

	
	return 0;
}