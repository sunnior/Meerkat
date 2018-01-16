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
		Optimizer(ComputeType type)
			: m_type(type)
		{ };

		virtual ~Optimizer() {};

		virtual void Update(const dl_vector<::std::pair<Tensor*, Tensor*>>& params) = 0;
	protected:
		ComputeType		m_type;
	};
}

#endif