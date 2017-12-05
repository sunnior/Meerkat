#include "ClassNllCriterion.h"

namespace DeepLearning
{
	void ClassNllCriterion::_ForwardCpu(const Tensor* input, const Tensor* target, dl_tensor* output)
	{
		dl_uint32 batch_size = input->GetShape(0);
		dl_uint32 label_size = input->GetShape(1);
		const dl_tensor* input_data = input->GetData();
		const dl_tensor* target_data = target->GetData();

		dl_tensor loss = 0;
		for (dl_uint32 i = 0; i < batch_size; ++i)
		{
			loss += input_data[i*label_size + (dl_uint32)target_data[i]];
		}
		*output = loss / batch_size;
	}

	void ClassNllCriterion::_ForwardGpu(const Tensor* input, const Tensor* target, dl_tensor* output)
	{

	}

	void ClassNllCriterion::_BackwardCpu(const Tensor* input, const Tensor* target, Tensor* output)
	{

	}

	void ClassNllCriterion::_BackwardGpu(const Tensor* input, const Tensor* target, Tensor* output)
	{

	}

}