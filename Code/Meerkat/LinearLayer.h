#ifndef __MEERKAT_LINEARLAYER_H__
#define __MEERKAT_LINEARLAYER_H__

#include "Layer.h"
#include "Tensor.h"

namespace Meerkat
{
	class LinearLayer : public Layer
	{
	public:
		LinearLayer(dl_uint32 input_num, dl_uint32 output_num);
		~LinearLayer();

		void InitData(ComputeType type);
		void Forward(Tensor* input, Tensor* output)
		{
			(m_compute_type == ComputeType_CPU) ? _ForwardCpu(input, output) : _ForwardGpu(input, output);
		}
	private:
		void _ForwardCpu(Tensor* input, Tensor* output);
		void _ForwardGpu(Tensor* input, Tensor* output);
	private:
		Tensor* m_weight{ nullptr };
		Tensor* m_bias{ nullptr };
	};
}

#endif