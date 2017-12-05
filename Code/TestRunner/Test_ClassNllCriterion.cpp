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
		dl_tensor correct_value = 10.5f;
		
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
}

