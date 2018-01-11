#include "UnitTest++.h"
#include "Layers/LogSoftMaxLayer.h"
#include "Util.h"

using namespace DeepLearning;

SUITE(LogSoftMaxLayer)
{
	TEST(Forward)
	{
		dl_tensor input_value[2][4] = { { 1, 2, 3, 4 },{ 0.1f, 0.1f, 0.1f, 0.1f } };
		dl_tensor target_value[2][4] = { { -3.44018969f, -2.44018969f, -1.44018969f, -0.44018969f },{ -1.38629436f, -1.38629436f, -1.38629436f, -1.38629436f } };
		LogSoftMaxLayer* layer = DL_NEW(LogSoftMaxLayer)(ComputeType_CPU);

		Tensor* input = DL_NEW(Tensor)(ComputeType_CPU, { 2, 4 });
		Tensor* output = DL_NEW(Tensor)(ComputeType_CPU, { 2, 4 });
		Tensor* target = DL_NEW(Tensor)(ComputeType_CPU, { 2, 4 });

		input->LoadData((dl_tensor*)input_value);
		target->LoadData((dl_tensor*)target_value);

		layer->Forward(input, output);


		CHECK(dl_check_cpumem_close(target, output));

		DL_SAFE_DELETE(input);
		DL_SAFE_DELETE(output);
		DL_SAFE_DELETE(target);
		DL_SAFE_DELETE(layer);
	}

	TEST(Backward)
	{
		dl_tensor input_value[2][4] = { { 1, 2, 3, 4 },{ 0.1f, 0.1f, 0.2f, 0.1f } };
		dl_tensor grad_input_value[2][4] = { { 0, 0, 0.5, 0 },{ 0.5f, 0, 0, 0 } };
		dl_tensor grad_output_value[2][4] = { { 0, 0, 0.3815585f, 0 },{ 0.3782023f, 0, 0, 0 } };

		LogSoftMaxLayer* layer = DL_NEW(LogSoftMaxLayer)(ComputeType_CPU);
		Tensor* input = DL_NEW(Tensor)(ComputeType_CPU, { 2, 4 });
		Tensor* grad_input = DL_NEW(Tensor)(ComputeType_CPU, { 2, 4 });
		Tensor* grad_output = DL_NEW(Tensor)(ComputeType_CPU, { 2, 4 });
		Tensor* target_grad_output = DL_NEW(Tensor)(ComputeType_CPU, { 2, 4 });

		input->LoadData((dl_tensor*)input_value);
		grad_input->LoadData((dl_tensor*)grad_input_value);
		target_grad_output->LoadData((dl_tensor*)grad_output_value);

		layer->Backward(input, grad_input, grad_output);

		CHECK(dl_check_cpumem_close(target_grad_output, grad_output));

		DL_SAFE_DELETE(input);
		DL_SAFE_DELETE(grad_output);
		DL_SAFE_DELETE(grad_input);
		DL_SAFE_DELETE(target_grad_output);
		DL_SAFE_DELETE(layer);
	}

}

