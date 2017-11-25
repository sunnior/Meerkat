#include "UnitTest++.h"
#include "Criterions/MeanSquareErrorCriterion.h"
#include "Util.h"



using namespace DeepLearning;



TEST(MeanSquareErrorCriterion)

{
	const dl_uint32 batch_size = 8;
	dl_tensor input_value[batch_size] = { 0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 2.0f, 3.0f, 4.0f };
	dl_tensor target_value[batch_size] = { 0.4f, 0.3f, 0.2f, 0.1f, 0.0f, 100.0f, 1.0f, 1.0f};
	dl_tensor loss_value = 1202.275f;

	Tensor* input = DL_NEW(Tensor) { batch_size };
	Tensor* target = DL_NEW(Tensor) { batch_size };
	MeanSquareErrorCriterion* MseCriterion = DL_NEW(MeanSquareErrorCriterion)(batch_size);

	dl_tensor output;

	input->Alloc(ComputeType_CPU);
	target->Alloc(ComputeType_CPU);

	input->LoadData(ComputeType_CPU, (dl_tensor*)input_value);
	target->LoadData(ComputeType_CPU, (dl_tensor*)target_value);

	MseCriterion->Alloc(ComputeType_CPU);
	MseCriterion->Forward(ComputeType_CPU, input, target, &output);

	CHECK_CLOSE(loss_value, output, 0.00001f);

	DL_SAFE_DELETE(input);
	DL_SAFE_DELETE(target);
	DL_SAFE_DELETE(MseCriterion);
}