#include "UnitTest++.h"
#include "Layers/LinearLayer.h"
#include "Util.h"



using namespace DeepLearning;



SUITE(LinearLayer)
{
	TEST(Forward)
	{
		dl_tensor input_value[2][4] = { { 0.1f, 0.2f, 0.3f, 0.4f },{ 1.0f, 2.0f, 3.0f, 4.0f } };
		dl_tensor weight_value[2][4] = { { 1.0f, 2.0f, 3.0f, 4.0f },{ 0.0f, 100.0f, 1.0f, 1.0f } };
		dl_tensor bias_value[2] = { 0.33f, 0.43f };
		dl_tensor target_value[2][2] = { { 3.33f, 21.13f },{ 30.33f, 207.43f } };

		Tensor* input = DL_NEW(Tensor)(ComputeType_CPU, { 2, 4 });
		Tensor* output = DL_NEW(Tensor)(ComputeType_CPU, { 2, 2 });
		Tensor* target = DL_NEW(Tensor)(ComputeType_CPU, { 2, 2 });
		LinearLayer* linear_layer = DL_NEW(LinearLayer)(ComputeType_CPU, true, 4, 2);

		input->LoadData((dl_tensor*)input_value);
		target->LoadData((dl_tensor*)target_value);

		linear_layer->GetWeight()->LoadData((dl_tensor*)weight_value);
		linear_layer->GetBias()->LoadData((dl_tensor*)bias_value);

		linear_layer->Forward(input, output);

		CHECK(dl_check_cpumem_close(target, output));

		DL_SAFE_DELETE(target);
		DL_SAFE_DELETE(output);
		DL_SAFE_DELETE(input);
		DL_SAFE_DELETE(linear_layer);
	}

	TEST(Backward)
	{
		dl_tensor input_value[2][2] = { { 1.0f, 1.0f },{ 2.0f, 3.0f } };
		dl_tensor grad_input_value[2][3] = { { 1.0f, 2.0f, 3.0f },{ 4.0f, 5.0f, 6.0f } };
		dl_tensor weight_value[3][2] = { { 1.0f, 4.0f },{ 2.0f, 5.0f },{ 3.0f , 6.0f } };
		dl_tensor bias_value[3] = { 0.1f, 0.2f, 0.3f };

		dl_tensor correct_grad_weight_value[3][2] = { { 9.0f, 13.0f },{ 12.0f, 17.0f },{ 15.0f, 21.0f } };
		dl_tensor correct_grad_output_value[2][2] = { { 14.0f, 32.0f },{ 32.0f, 77.0f } };
		dl_tensor correct_grad_bias_value[3] = { 5.0f, 7.0f, 9.0f };

		Tensor* input = DL_NEW(Tensor)(ComputeType_CPU, { 2, 2 });
		Tensor* grad_input = DL_NEW(Tensor)(ComputeType_CPU, { 2, 3 });
		Tensor* grad_output = DL_NEW(Tensor)(ComputeType_CPU, { 2, 2 });
		Tensor* correct_grad_weight = DL_NEW(Tensor)(ComputeType_CPU, { 2, 3 });
		Tensor* correct_grad_output = DL_NEW(Tensor)(ComputeType_CPU, { 2, 2 });
		Tensor* correct_grad_bias = DL_NEW(Tensor)(ComputeType_CPU, { 3 });
		LinearLayer* linear_layer = DL_NEW(LinearLayer)(ComputeType_CPU, true, 2, 3);

		input->LoadData((dl_tensor*)input_value);
		grad_input->LoadData((dl_tensor*)grad_input_value);
		correct_grad_weight->LoadData((dl_tensor*)correct_grad_weight_value);
		correct_grad_output->LoadData((dl_tensor*)correct_grad_output_value);
		correct_grad_bias->LoadData((dl_tensor*)correct_grad_bias_value);

		linear_layer->GetWeight()->LoadData((dl_tensor*)weight_value);
		linear_layer->GetBias()->LoadData((dl_tensor*)bias_value);

		linear_layer->Backward(input, grad_input, grad_output);

		CHECK(dl_check_cpumem_close(correct_grad_weight, linear_layer->GetGradWeight()));
		CHECK(dl_check_cpumem_close(correct_grad_output, grad_output));
		CHECK(dl_check_cpumem_close(correct_grad_bias, linear_layer->GetGradBias()));

		DL_SAFE_DELETE(input);
		DL_SAFE_DELETE(grad_input);
		DL_SAFE_DELETE(grad_output);
		DL_SAFE_DELETE(correct_grad_weight);
		DL_SAFE_DELETE(correct_grad_bias);
		DL_SAFE_DELETE(correct_grad_output);
		DL_SAFE_DELETE(linear_layer);

	}
}