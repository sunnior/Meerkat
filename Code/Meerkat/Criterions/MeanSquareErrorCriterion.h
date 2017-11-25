#ifndef __MEERKAT_MIN_SQUARE_ERROR_CRITERION_H__
#define __MEERKAT_MIN_SQUARE_ERROR_CRITERION_H__

#include "Criterion.h"

namespace DeepLearning
{
	class MeanSquareErrorCriterion : public Criterion
	{
	public:
		MeanSquareErrorCriterion(ComputeType type, dl_uint32 batch_size);
		~MeanSquareErrorCriterion();

		void _ForwardCpu(const Tensor* input, const Tensor* target, dl_tensor* output) override;
		void _ForwardGpu(const Tensor* input, const Tensor* target, dl_tensor* output) override;

	private:
		Tensor* m_internal;
	};
}

#endif