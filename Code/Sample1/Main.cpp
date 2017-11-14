#include "LinearLayer.h"

using namespace Meerkat;

int main()
{
	Tensor* input = DL_NEW(Tensor) { 2 };
	Tensor* output = DL_NEW(Tensor) { 1 };
	input->Alloc(ComputeType_CPU);
	output->Alloc(ComputeType_CPU);
	input->Zeros();

	LinearLayer* linear_layer = DL_NEW(LinearLayer)(2, 1);
	linear_layer->InitData(ComputeType_CPU);
	linear_layer->Forward(input, output);

	
	return 0;
}