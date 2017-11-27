#include "UnitTest++.h"
#include "Criterions/MeanSquareErrorCriterion.h"
#include "Util.h"



using namespace DeepLearning;



SUITE(MeanSquareErrorCriterion)
{
	TEST(Forward)
	{
		const dl_uint32 batch_size = 8;
		dl_tensor input_value[batch_size] = { 0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 2.0f, 3.0f, 4.0f };
		dl_tensor target_value[batch_size] = { 0.4f, 0.3f, 0.2f, 0.1f, 0.0f, 100.0f, 1.0f, 1.0f };
		dl_tensor loss_value = 1202.275f;

		Tensor* input = DL_NEW(Tensor)(ComputeType_CPU, { batch_size });
		Tensor* target = DL_NEW(Tensor)(ComputeType_CPU, { batch_size });
		MeanSquareErrorCriterion* MseCriterion = DL_NEW(MeanSquareErrorCriterion)(ComputeType_CPU, batch_size);

		dl_tensor output;

		input->LoadData((dl_tensor*)input_value);
		target->LoadData((dl_tensor*)target_value);

		MseCriterion->Forward(input, target, &output);

		CHECK_CLOSE(loss_value, output, 0.00001f);

		DL_SAFE_DELETE(input);
		DL_SAFE_DELETE(target);
		DL_SAFE_DELETE(MseCriterion);
	}

	TEST(Backward)
	{
		const dl_uint32 batch_size = 8;
		dl_tensor input_value[batch_size] = { 0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 2.0f, 3.0f, 4.0f };
		dl_tensor target_value[batch_size] = { 0.4f, 0.3f, 0.2f, 0.1f, 0.0f, 100.0f, 1.0f, 1.0f };
		dl_tensor correct_value[batch_size] = { -0.075f, -0.025f, 0.025f, 0.075f, 0.25f, -24.5f, 0.5f, 0.75f };

		Tensor* input = DL_NEW(Tensor)(ComputeType_CPU, { batch_size });
		Tensor* target = DL_NEW(Tensor)(ComputeType_CPU, { batch_size });
		Tensor* output = DL_NEW(Tensor)(ComputeType_CPU, { batch_size });
		Tensor* correct = DL_NEW(Tensor)(ComputeType_CPU, { batch_size });
		MeanSquareErrorCriterion* MseCriterion = DL_NEW(MeanSquareErrorCriterion)(ComputeType_CPU, batch_size);

		input->LoadData((dl_tensor*)input_value);
		target->LoadData((dl_tensor*)target_value);
		correct->LoadData((dl_tensor*)correct_value);

		MseCriterion->Backward(input, target, output);

		CHECK(dl_check_cpumem_close(correct, output));
		DL_SAFE_DELETE(input);
		DL_SAFE_DELETE(target);
		DL_SAFE_DELETE(output);
		DL_SAFE_DELETE(MseCriterion);
	}

}