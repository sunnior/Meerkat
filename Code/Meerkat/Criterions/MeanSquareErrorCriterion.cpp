#include "MeanSquareErrorCriterion.h"
#include "Common/Blas.h"

namespace DeepLearning
{

	MeanSquareErrorCriterion::MeanSquareErrorCriterion(ComputeType type, dl_uint32 batch_size)
		: Criterion(type)
	{
		m_internal = DL_NEW(Tensor)(type, { batch_size });
	}

	MeanSquareErrorCriterion::~MeanSquareErrorCriterion()
	{
		DL_SAFE_DELETE(m_internal);
	}

	void MeanSquareErrorCriterion::_ForwardCpu(const Tensor* input, const Tensor* target, dl_tensor* output)
	{
		dl_uint32 batch_size = target->GetShape(0);
		
		dl_tensor* internal_data = m_internal->GetData();
		const dl_tensor* input_data = input->GetData();
		const dl_tensor* target_data = target->GetData();
		
		dl_copy_cpu<dl_tensor>(batch_size, target_data, 1, internal_data, 1);
		dl_axpy_cpu<dl_tensor>(batch_size, -1.0f, input_data, 1, internal_data, 1);
		*output = dl_dot_cpu<dl_tensor>(batch_size, internal_data, 1, internal_data, 1);
		*output /= batch_size;
	}

	void MeanSquareErrorCriterion::_ForwardGpu(const Tensor* input, const Tensor* target, dl_tensor* output)
	{

	}

	void MeanSquareErrorCriterion::_BackwardCpu(const Tensor* input, const Tensor* target, Tensor* output)
	{
		dl_uint32 batch_size = target->GetShape(0);

		dl_tensor* output_data = output->GetData();
		const dl_tensor* input_data = input->GetData();
		const dl_tensor* target_data = target->GetData();

		dl_tensor norm = 2.0f / batch_size;
		dl_copy_cpu<dl_tensor>(batch_size, input_data, 1, output_data, 1);
		dl_axpy_cpu<dl_tensor>(batch_size, -1.0f, target_data, 1, output_data, 1);
		dl_scal_cpu<dl_tensor>(batch_size, norm, output_data, 1);
	}

	void MeanSquareErrorCriterion::_BackwardGpu(const Tensor* input, const Tensor* target, Tensor* output)
	{

	}

}