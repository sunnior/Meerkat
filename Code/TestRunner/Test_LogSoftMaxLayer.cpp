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
	}

	TEST(Backward)
	{
	}

}

