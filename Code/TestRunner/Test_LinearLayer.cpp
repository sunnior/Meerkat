#include "UnitTest++.h"
#include "LinearLayer.h"
#include "Util.h"



using namespace DeepLearning;



TEST(LinearLayer)

{

	dl_tensor input_value[2][4] = { { 0.1f, 0.2f, 0.3f, 0.4f },{ 1.0f, 2.0f, 3.0f, 4.0f } };

	dl_tensor weight_value[2][4] = { { 1.0f, 2.0f, 3.0f, 4.0f },{ 0.0f, 100.0f, 1.0f, 1.0f } };

	dl_tensor bias_value[2] = { 0.33f, 0.43f };

	dl_tensor target_value[2][2] = { { 3.33f, 21.13f },{ 30.33f, 207.43f } };



	Tensor* input = DL_NEW(Tensor) { 2, 4 };

	Tensor* output = DL_NEW(Tensor) { 2, 2 };

	Tensor* target = DL_NEW(Tensor) { 2, 2 };

	input->Alloc(ComputeType_CPU);

	output->Alloc(ComputeType_CPU);

	target->Alloc(ComputeType_CPU);



	input->LoadData(ComputeType_CPU, (dl_tensor*)input_value);

	target->LoadData(ComputeType_CPU, (dl_tensor*)target_value);





	LinearLayer* linear_layer = DL_NEW(LinearLayer)(4, 2);

	linear_layer->Alloc(ComputeType_CPU);

	linear_layer->GetWeight()->LoadData(ComputeType_CPU, (dl_tensor*)weight_value);

	linear_layer->GetBias()->LoadData(ComputeType_CPU ,(dl_tensor*)bias_value);



	linear_layer->Forward(ComputeType_CPU, input, output);



	CHECK(dl_check_cpumem_close(target, output));

}