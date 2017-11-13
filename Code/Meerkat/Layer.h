#ifndef __MEERKAT_LAYER_H__
#define __MEERKAT_LAYER_H__

#include "Platform.h"

namespace Meerkat
{
	class Layer
	{
	public:
		Layer(ComputeType type)
			: m_compute_type(type)
		{};


		virtual ~Layer() {};

	protected:
		ComputeType m_compute_type{ ComputeType_CPU };
	};
}
#endif