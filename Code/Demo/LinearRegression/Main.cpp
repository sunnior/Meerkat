#include "Layers/LinearLayer.h"
#include "Common/Meerkat.h"

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

	dl_tensor input_data[10][2] = {
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

	dl_tensor target_data[10] = {
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

	Tensor* input = DL_NEW(Tensor)(ComputeType_CPU, { 10, 2 });
	Tensor* target = DL_NEW(Tensor)(ComputeType_CPU, { 10 });
	Tensor* output = DL_NEW(Tensor)(ComputeType_CPU, { 10 });
	input->LoadData((dl_tensor*)input_data);
	target->LoadData((dl_tensor*)target_data);

	LinearLayer* linear_layer = DL_NEW(LinearLayer)(ComputeType_CPU, true, 2, 1);
	linear_layer->Forward(input, output);

	
	return 0;
}