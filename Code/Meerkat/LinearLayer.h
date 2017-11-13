#ifndef __MEERKAT_LINEARLAYER_H__
#define __MEERKAT_LINEARLAYER_H__

#include "Layer.h"
#include "Tensor.h"

namespace Meerkat
{
	class LinearLayer : public Layer
	{
	public:
		LinearLayer(ComputeType type, dl_uint32 input_num, dl_uint32 output_num);
		~LinearLayer();
	private:
		Tensor* m_weight{ nullptr };
		Tensor* m_bias{ nullptr };
	};
}

#endif