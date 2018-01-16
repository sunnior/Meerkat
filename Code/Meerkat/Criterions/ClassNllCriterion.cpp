#include "ClassNllCriterion.h"

namespace DeepLearning
{
	void ClassNllCriterion::_ForwardCpu(const Tensor* input, const Tensor* target, dl_tensor* output)
	{
		dl_uint32 batch_size = input->GetShape(0);
		dl_uint32 label_size = input->GetShape(1);
		const dl_tensor* input_data = input->GetData();
		const dl_tensor* target_data = target->GetData();

		*output = 0;
		for (dl_uint32 i = 0; i < batch_size; ++i)
		{
			*output -= input_data[i*label_size + (dl_uint32)target_data[i]];
		}
	}

	void ClassNllCriterion::_ForwardGpu(const Tensor* input, const Tensor* target, dl_tensor* output)
	{

	}

	void ClassNllCriterion::_BackwardCpu(const Tensor* input, const Tensor* target, Tensor* output)
	{
		dl_uint32 batch_size = input->GetShape(0);
		dl_uint32 param_size = input->GetShape(1);

		TODO("where to zero output");
	    output->Zeros();

		dl_tensor* output_data = output->GetData();
		const dl_tensor* target_data = target->GetData();

		dl_tensor value = -1.0f;

		for (dl_uint32 i = 0; i < batch_size; ++i)
		{
			*(output_data + i*param_size + (dl_uint32)target_data[i]) = value;
		}
	}

	void ClassNllCriterion::_BackwardGpu(const Tensor* input, const Tensor* target, Tensor* output)
	{

	}

}