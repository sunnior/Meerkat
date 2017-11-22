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

	void LinearLayer::Alloc(ComputeType type)
	{
		m_weight->Alloc(type);
		m_bias->Alloc(type);
		TODO("way to init");
		m_weight->Zeros();
		m_bias->Zeros();
	}

	void LinearLayer::_ForwardCpu(Tensor* input, Tensor* output)
	{
		TODO("check dimension");
		dl_uint32 d1 = m_weight->GetShape(0);
		dl_uint32 d2 = m_weight->GetShape(1);

		output->Zeros();

		dl_tensor* input_data = input->GetCpuData();
		dl_tensor* output_data = output->GetCpuData();
		dl_tensor* weight_data = m_weight->GetCpuData();
		dl_tensor* bias_data = m_bias->GetCpuData();

		for (dl_uint32 i = 0; i < d2; ++i)
		{
			for (dl_uint32 j = 0; j < d1; ++j)
			{
				output_data[i] += input_data[i*d1 + j] * weight_data[i*d1 + j];
			}
			output_data[i] += bias_data[i];
		}
	}

	void LinearLayer::_ForwardGpu(Tensor* input, Tensor* output)
	{

	}

}
