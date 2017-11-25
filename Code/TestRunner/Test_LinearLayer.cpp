#include "UnitTest++.h"
#include "Layers/LinearLayer.h"
#include "Util.h"



using namespace DeepLearning;



TEST(LinearLayer)
{
	dl_tensor input_value[2][4] = { { 0.1f, 0.2f, 0.3f, 0.4f },{ 1.0f, 2.0f, 3.0f, 4.0f } };
	dl_tensor weight_value[2][4] = { { 1.0f, 2.0f, 3.0f, 4.0f },{ 0.0f, 100.0f, 1.0f, 1.0f } };
	dl_tensor bias_value[2] = { 0.33f, 0.43f };
	dl_tensor target_value[2][2] = { { 3.33f, 21.13f },{ 30.33f, 207.43f } };

	Tensor* input = DL_NEW(Tensor)(ComputeType_CPU, { 2, 4 });
	Tensor* output = DL_NEW(Tensor)(ComputeType_CPU, { 2, 2 });
	Tensor* target = DL_NEW(Tensor)(ComputeType_CPU, { 2, 2 });
	LinearLayer* linear_layer = DL_NEW(LinearLayer)(ComputeType_CPU, 4, 2);

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