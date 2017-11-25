#include "MeanSquareErrorCriterion.h"
#include "Common/Blas.h"

namespace DeepLearning
{

	MeanSquareErrorCriterion::MeanSquareErrorCriterion(dl_uint32 batch_size)
	{
		m_internal = DL_NEW(Tensor) { batch_size };
	}

	MeanSquareErrorCriterion::~MeanSquareErrorCriterion()
	{
		DL_SAFE_DELETE(m_internal);
	}

	void MeanSquareErrorCriterion::Alloc(ComputeType type)
	{
		m_internal->Alloc(type);
	}

	void MeanSquareErrorCriterion::_ForwardCpu(const Tensor* input, const Tensor* target, dl_tensor* output)
	{
		dl_uint32 batch_size = target->GetShape(0);
		
		dl_tensor* internal_data = m_internal->GetCpuData();
		const dl_tensor* input_data = input->GetCpuData();
		const dl_tensor* target_data = target->GetCpuData();
		
		dl_copy_cpu<dl_tensor>(batch_size, target_data, 1, internal_data, 1);
		dl_axpy_cpu<dl_tensor>(batch_size, -1.0f, input_data, 1, internal_data, 1);
		*output = dl_dot_cpu<dl_tensor>(batch_size, internal_data, 1, internal_data, 1);
		*output /= batch_size;
	}

	void MeanSquareErrorCriterion::_ForwardGpu(const Tensor* input, const Tensor* target, dl_tensor* output)
	{

	}

}