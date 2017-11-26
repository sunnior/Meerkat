#ifndef __MEERKAT_CRITERION_H__
#define __MEERKAT_CRITERION_H__

#include "Common/Tensor.h"

namespace DeepLearning
{
	class Criterion
	{
	public:
		Criterion(ComputeType type)
			: m_type(type)
		{};

		virtual ~Criterion() {};

		void Forward(const Tensor* input, const Tensor* target, dl_tensor* output)
		{
			(m_type == ComputeType_CPU) ? _ForwardCpu(input, target, output) : _ForwardGpu(input, target, output);
		}

		void Backward(const Tensor* input, const Tensor* target, Tensor* output)
		{
			(m_type == ComputeType_CPU) ? _BackwardCpu(input, target, output) : _BackwardGpu(input, target, output);
		}

	protected:
		virtual void _ForwardCpu(const Tensor* input, const Tensor* target, dl_tensor* output) = 0;
		virtual void _ForwardGpu(const Tensor* input, const Tensor* target, dl_tensor* output) = 0;

		virtual void _BackwardCpu(const Tensor* input, const Tensor* target, Tensor* output) = 0;
		virtual void _BackwardGpu(const Tensor* input, const Tensor* target, Tensor* output) = 0;


	protected:
		ComputeType m_type;
	};
}

#endif