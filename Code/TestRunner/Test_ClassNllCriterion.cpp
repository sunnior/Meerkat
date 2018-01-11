#include "UnitTest++.h"
#include "Criterions/ClassNllCriterion.h"
#include "Util.h"

using namespace DeepLearning;

SUITE(ClassNllCriterion)
{
	TEST(Forward)
	{
		dl_tensor target_value[2] = { 0, 1 };
		dl_tensor input_value[2][2] = { {1, 2}, {10, 20} };
		dl_tensor correct_value = 21.0f;
		
		Tensor* target = DL_NEW(Tensor)(ComputeType_CPU, { 2 });
		Tensor* input = DL_NEW(Tensor)(ComputeType_CPU, { 2, 2 });
		ClassNllCriterion* criterion = DL_NEW(ClassNllCriterion)(ComputeType_CPU);

		target->LoadData((dl_tensor*)target_value);
		input->LoadData((dl_tensor*)input_value);

		dl_tensor loss;
		criterion->Forward(input, target, &loss);

		CHECK_EQUAL(correct_value, loss);

		DL_SAFE_DELETE(target);
		DL_SAFE_DELETE(input);
		DL_SAFE_DELETE(criterion);
	}

	TEST(Backward)
	{
		dl_tensor target_value[2] = { 0, 1 };
		dl_tensor correct_grad_value[2][2] = { { -1.0f, 0.0f } ,{ 0.0f, -1.0f } };
		Tensor* target = DL_NEW(Tensor)(ComputeType_CPU, { 2 });
		Tensor* grad_output = DL_NEW(Tensor)(ComputeType_CPU, { 2, 2 });
		Tensor* input = DL_NEW(Tensor)(ComputeType_CPU, { 2, 2 });
		Tensor* correct_grad_output = DL_NEW(Tensor)(ComputeType_CPU, { 2, 2 });
		ClassNllCriterion* criterion = DL_NEW(ClassNllCriterion)(ComputeType_CPU);

		target->LoadData((dl_tensor*)target_value);
		correct_grad_output->LoadData((dl_tensor*)correct_grad_value);
		criterion->Backward(input, target, grad_output);

		CHECK(dl_check_cpumem_close(correct_grad_output, grad_output));
	}
}

