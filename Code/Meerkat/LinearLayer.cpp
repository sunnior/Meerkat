#include "LinearLayer.h"

namespace Meerkat
{
	LinearLayer::LinearLayer(ComputeType type, dl_uint32 input_num, dl_uint32 output_num)
		: Layer(type)
	{
		dl_uint32 weight_shape[2];
		weight_shape[0] = input_num;
		weight_shape[1] = output_num;

		m_weight = DL_NEW(Tensor)(weight_shape, 2);
		m_weight->Alloc(m_compute_type);

		dl_uint32 bias_shape = output_num;

		m_bias = DL_NEW(Tensor)(&bias_shape, 1);
		m_bias->Alloc(m_compute_type);
	}

	LinearLayer::~LinearLayer()
	{
		DL_SAFE_DELETE(m_weight);
		DL_SAFE_DELETE(m_bias);
	}
}
