#include "LinearLayer.h"

namespace Meerkat
{
	LinearLayer::LinearLayer(dl_uint32 input_num, dl_uint32 output_num)
	{
		m_weight = DL_NEW(Tensor){input_num, output_num};
		m_bias = DL_NEW(Tensor){output_num};
	}

	LinearLayer::~LinearLayer()
	{
		DL_SAFE_DELETE(m_weight);
		DL_SAFE_DELETE(m_bias);
	}

	void LinearLayer::InitData(ComputeType type)
	{
		m_weight->Alloc(m_compute_type);
		m_bias->Alloc(m_compute_type);
	}

	void LinearLayer::_ForwardCpu(Tensor* input, Tensor* output)
	{
		TODO("check dimension");

	}

	void LinearLayer::_ForwardGpu(Tensor* input, Tensor* output)
	{

	}

}
