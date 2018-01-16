#ifndef __MEERKAT_OPTIMIZER_H__
#define __MEERKAT_OPTIMIZER_H__

#include "Common/Platform.h"
#include "Util/dl_stl.h"

namespace DeepLearning
{
	class Tensor;

	class Optimizer
	{
	public:
		Optimizer(ComputeType type, dl_vector<Tensor*> params, dl_vector<Tensor*> grad_params)
			: m_type(type)
			, m_params(params)
			, m_grad_params(grad_params)
		{ };

		virtual ~Optimizer() {};

		virtual void Update() = 0;
	protected:
		ComputeType		m_type;
		dl_vector<Tensor*>		m_params;
		dl_vector<Tensor*>		m_grad_params;
	};
}

#endif