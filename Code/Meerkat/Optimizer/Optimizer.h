#ifndef __MEERKAT_OPTIMIZER_H__
#define __MEERKAT_OPTIMIZER_H__

#include "Common/Platform.h"

namespace DeepLearning
{
	class Tensor;

	class Optimizer
	{
	public:
		Optimizer(ComputeType type, Tensor** params, Tensor** grad_params, dl_uint32 count)
			: m_type(type)
			, m_params(params)
			, m_grad_params(grad_params)
			, m_count(count)
		{ };

		virtual ~Optimizer() {};

		virtual void Update() = 0;
	protected:
		const dl_uint32 m_count;
		ComputeType		m_type;
		Tensor**		m_params;
		Tensor**		m_grad_params;
	};
}

#endif