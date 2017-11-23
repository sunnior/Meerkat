#include <Windows.h>
#include "LinearLayer.h"
#include "Meerkat.h"

using namespace DeepLearning;

int main()
{
	DeepLearning::Init();

	Tensor* input = DL_NEW(Tensor) { 1, 2 };
	Tensor* output = DL_NEW(Tensor) { 1, 1 };
	input->Alloc(ComputeType_CPU);
	output->Alloc(ComputeType_CPU);
	input->Zeros();

	LinearLayer* linear_layer = DL_NEW(LinearLayer)(2, 1);
	linear_layer->Alloc(ComputeType_CPU);
	linear_layer->Forward(ComputeType_CPU, input, output);

	
	return 0;
}