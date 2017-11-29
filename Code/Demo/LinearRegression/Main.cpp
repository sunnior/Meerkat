#include "Common/Meerkat.h"
#include "Layers/LinearLayer.h"
#include "Optimizer/SgdOptimizer.h"
#include "Criterions/MeanSquareErrorCriterion.h"

using namespace DeepLearning;

int main()
{
	DeepLearning::Init();

/*
	data = torch.Tensor{
		{ 40,  6,  4 },
		{ 44, 10,  4 },
		{ 46, 12,  5 },
		{ 48, 14,  7 },
		{ 52, 16,  9 },
		{ 58, 18, 12 },
		{ 60, 22, 14 },
		{ 68, 24, 20 },
		{ 74, 26, 21 },
		{ 80, 32, 24 }
	}*/

	const dl_uint32 batch_size = 10;
	dl_tensor input_data[batch_size][2] = {
		//fertilizer, insecticide
			{ 6,  4 },
			{ 10,  4 },
			{ 12,  5 },
			{ 14,  7 },
			{ 16,  9 },
			{ 18, 12 },
			{ 22, 14 },
			{ 24, 20 },
			{ 26, 21 },
			{ 32, 24 }
	};

	dl_tensor target_data[batch_size] = {
		40,
		44,
		46,
		48,
		52,
		58,
		60,
		68,
		74,
		80,
	};

	Tensor* input = DL_NEW(Tensor)(ComputeType_CPU, { batch_size, 2 });
	Tensor* target = DL_NEW(Tensor)(ComputeType_CPU, { batch_size, 1 });
	Tensor* output = DL_NEW(Tensor)(ComputeType_CPU, { batch_size, 1 });
	Tensor* grad_output = DL_NEW(Tensor)(ComputeType_CPU, { batch_size, 1 });
	LinearLayer* linear_layer = DL_NEW(LinearLayer)(ComputeType_CPU, true, 2, 1);
	MeanSquareErrorCriterion* criterion = DL_NEW(MeanSquareErrorCriterion)(ComputeType_CPU, batch_size);
	dl_uint32 count;
	Tensor** params;
	Tensor** grad_params;
	linear_layer->GetLearnableParam(params, grad_params, count);
	Optimizer* optimizer = DL_NEW(SgdOptimizer)(ComputeType_CPU, params, grad_params, count);

	input->LoadData((dl_tensor*)input_data);
	target->LoadData((dl_tensor*)target_data);

	dl_tensor loss = 1000;
	while (loss > 0.2)
	{
		linear_layer->Forward(input, output);
		criterion->Forward(output, target, &loss);

		criterion->Backward(output, target, grad_output);
		linear_layer->Backward(input, grad_output, nullptr);

		optimizer->Update();
	}
	return 0;
}